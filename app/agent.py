"""LLM agent: turns a natural-language question into geometry tool calls.

The model is reached through any OpenAI-compatible endpoint, configured in
`app.config`. It can only request the whitelisted tools of `app.geometry`; it
never computes geometry and never writes or executes code.

Context engineering: the system prompt carries a short description of the
loaded models and of what the user selected in the viewer (active model,
picked points, current view). The LLM can therefore fill in tool arguments
on the first round-trip, without a discovery call.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .geometry import TOOL_SCHEMAS, VIEW_ACTION, Catalog, Point, describe_model, for_llm, run_tool

logger = logging.getLogger(__name__)

MAX_PICKED_POINTS = 10

INSTRUCTIONS = """\
You are a geometry assistant for 3D models (triangle meshes). Lengths are in \
millimetres and Z is up.

Rules:
- You never compute geometry yourself. Every number in your answer must come \
from a tool result of this conversation; never estimate or invent one.
- The context below only helps you fill in tool arguments. To report a \
quantity, call the tool that measures it, so the answer can be traced.
- Unless the user names another model, use the active model.
- "The points I clicked" are the picked points of the context, in order. If \
the question needs more points than were picked, ask the user to click on \
the model instead of calling a tool.
- If no tool can answer the question, say so.
- Keep the final answer short, with units."""

RETRYABLE = (RateLimitError, APIConnectionError, InternalServerError)
MAX_BACKOFF_S = 60.0


class Selection(BaseModel):
    """What the user selected in the viewer, sent with each question."""

    model: str | None = Field(default=None, max_length=64, description="Id of the model shown.")
    points: list[Point] = Field(
        default_factory=list,
        max_length=MAX_PICKED_POINTS,
        description="Points picked on the model, [x, y, z] in mm, in click order.",
    )
    view: Literal["top", "front", "side", "iso", "custom"] | None = Field(
        default=None, description="Current camera view; custom after a free rotation."
    )


def build_context(cat: Catalog, selection: Selection) -> str:
    """Describe the loaded models and the user's selection for the system prompt.

    Only identifiers and numbers produced by the application are included:
    model ids are slugs, so an uploaded file name cannot inject instructions.
    """
    lines = []
    active = selection.model if selection.model in cat.models else None
    if active:
        info = describe_model(active, cat.models[active])
        box = info["bounding_box_mm"]
        lines.append(
            f"Active model: {active} ({info['faces']} faces, bounding box from {box['min']} "
            f"to {box['max']} mm, {'watertight' if info['watertight'] else 'not watertight'})."
        )
    else:
        lines.append("Active model: none.")
    others = sorted(name for name in cat.models if name != active)
    if others:
        lines.append(f"Other models: {', '.join(others)}.")
    if selection.points:
        picked = "; ".join(f"P{i} = {point}" for i, point in enumerate(selection.points, start=1))
        lines.append(f"Picked points (mm): {picked}.")
    else:
        lines.append("Picked points: none.")
    if selection.view:
        lines.append(f"Current view: {selection.view}.")
    return "\n".join(lines)


def build_client(settings: Settings) -> OpenAI:
    """Create the OpenAI-compatible client; retries are handled by `_complete`."""
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=0,
    )


def _retry_delay(exc: Exception, attempt: int, base_s: float) -> float:
    """Seconds to wait before retrying: the server's Retry-After, else exponential backoff."""
    if isinstance(exc, APIStatusError):
        try:
            return min(float(exc.response.headers.get("retry-after", "")), MAX_BACKOFF_S)
        except ValueError:
            pass
    return min(base_s * 2**attempt, MAX_BACKOFF_S)


def _complete(client: OpenAI, settings: Settings, messages: list[dict[str, Any]]) -> Any:
    """Call the chat completion endpoint, retrying on rate limits and transient errors."""
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model, messages=messages, tools=TOOL_SCHEMAS
            )
            return response.choices[0].message
        except RETRYABLE as exc:
            if attempt == settings.llm_max_retries:
                raise
            delay = _retry_delay(exc, attempt, settings.llm_retry_base_s)
            logger.warning(
                "LLM call failed (%s); retry %d/%d in %.1f s",
                type(exc).__name__,
                attempt + 1,
                settings.llm_max_retries,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _parse_args(raw: str | None) -> dict[str, Any] | None:
    """Decode the JSON arguments of a tool call; None if they are not a JSON object."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def ask(
    question: str,
    cat: Catalog,
    selection: Selection | None = None,
    history: list[dict[str, str]] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the tool-calling loop and return the answer, its trace and the view actions.

    Args:
        question: The user's question in natural language.
        cat: Models the tools operate on.
        selection: Active model, picked points and current view of the viewer.
        history: Previous turns of the conversation, as
            `{"role": "user" | "assistant", "content": str}` messages.
        settings: Configuration override (defaults to the process settings).

    Returns:
        `{"answer": str, "trace": [{"tool", "args", "ok", "summary"}], "view_actions": [dict]}`
        where `view_actions` are to be applied by the 3D viewer, in order.
    """
    settings = settings or get_settings()
    client = build_client(settings)
    context = build_context(cat, selection or Selection())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{INSTRUCTIONS}\n\nContext:\n{context}"},
        *(history or []),
        {"role": "user", "content": question},
    ]
    trace: list[dict[str, Any]] = []
    view_actions: list[dict[str, Any]] = []

    for step in range(settings.max_steps):
        reply = _complete(client, settings, messages)

        if not reply.tool_calls:
            logger.info("Answered after %d step(s), %d tool call(s)", step + 1, len(trace))
            return {"answer": reply.content or "", "trace": trace, "view_actions": view_actions}

        messages.append(reply.model_dump(exclude_none=True))
        for call in reply.tool_calls:
            name = call.function.name
            args = _parse_args(call.function.arguments)
            if args is None:
                args, result = {}, {"error": "Tool arguments must be a JSON object."}
            else:
                result = run_tool(cat, name, args)
            logger.info("Tool call %s(%s)", name, args)
            ok = "error" not in result
            if not ok:
                logger.warning("Tool '%s' returned an error: %s", name, result["error"])
            trace.append(
                {
                    "tool": name,
                    "args": args,
                    "ok": ok,
                    "summary": result["summary"] if ok else result["error"],
                }
            )
            if VIEW_ACTION in result:
                view_actions.append(result[VIEW_ACTION])
            # The LLM gets compact numbers; face indices and polylines go to the viewer only.
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(for_llm(result))}
            )

    logger.warning("Stopped after %d steps without a final answer", settings.max_steps)
    return {
        "answer": "Stopped: the question needed too many steps.",
        "trace": trace,
        "view_actions": view_actions,
    }
