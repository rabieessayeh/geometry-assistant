"""FastAPI service exposing the geometry assistant, the raw tools and the 3D viewer.

Run:  uvicorn app.api:app --reload   (viewer on http://localhost:8000/)
"""

from __future__ import annotations

import logging
import math
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError, RateLimitError
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from .agent import Selection, ask
from .config import Settings, configure_logging, get_settings
from .geometry import (
    MESH_FILE_TYPES,
    TOOLS,
    Catalog,
    MeshError,
    describe_model,
    load_mesh,
    run_tool,
    to_stl,
)
from .ratelimit import RateLimiter
from .sessions import SessionStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
GLOBAL_KEY = "*"
SESSION_COOKIE = "ga_session"
SESSION_ID = re.compile(r"[A-Za-z0-9_-]{20,64}")
SESSION_MAX_AGE_S = 7 * 24 * 3600
FORM_OVERHEAD_BYTES = 4096  # multipart boundaries and headers around the uploaded file
LOCAL_HOSTS = ("localhost", "127.0.0.1")

UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


class Turn(BaseModel):
    """One previous message of the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class Question(BaseModel):
    """Body of `POST /ask`."""

    question: str = Field(min_length=1, max_length=1000)
    selection: Selection = Field(default_factory=Selection)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class ToolCall(BaseModel):
    """Body of `POST /tools`."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """One tool call made while answering a question."""

    tool: str
    args: dict[str, Any]
    ok: bool
    summary: str


class Answer(BaseModel):
    """Response of `POST /ask`."""

    answer: str
    trace: list[TraceStep]
    view_actions: list[dict[str, Any]]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box, in mm."""

    min: list[float]
    max: list[float]
    size: list[float]


class ModelSummary(BaseModel):
    """A model the tools can work on."""

    id: str
    source: Literal["sample", "upload"]
    faces: int
    vertices: int
    bounding_box_mm: BoundingBox
    watertight: bool


def _client_ip(request: Request, trust_forwarded_for: bool) -> str:
    """Address used as rate-limit key.

    Behind a reverse proxy every request comes from the proxy, so the first
    entry of X-Forwarded-For is used instead. A client can forge that header,
    which is why the per-client limit is backed by a global one.
    """
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _session_id(request: Request) -> str | None:
    """Session id of the request's cookie, if it has a well-formed one."""
    value = request.cookies.get(SESSION_COOKIE, "")
    return value if SESSION_ID.fullmatch(value) else None


def _wait_text(seconds: float) -> str:
    """Human-readable waiting time, rounded up to the minute."""
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes} minute{'s' if minutes > 1 else ''}"


def _window_text(seconds: int) -> str:
    """Human-readable rate-limit window: 'hour', '2 hours', '10 minutes'."""
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        if seconds % size == 0:
            count = seconds // size
            return unit if count == 1 else f"{count} {unit}s"
    return f"{seconds} seconds"


def create_app(settings: Settings | None = None, catalog: Catalog | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Configuration override (defaults to environment / `.env`).
        catalog: Pre-loaded sample models; when omitted they are read from
            `settings.samples_dir` at startup.
    """
    settings = settings or get_settings()
    llm_is_local = any(host in settings.llm_base_url for host in LOCAL_HOSTS)
    llm_configured = llm_is_local or not settings.llm_key_is_placeholder

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        app.state.samples = catalog or Catalog.from_folder(settings.samples_dir)
        if not llm_configured:
            logger.warning("LLM_API_KEY is not set: /ask is disabled, /tools still works")
        logger.info(
            "Ready: %d sample model(s), LLM model '%s' at %s",
            len(app.state.samples.models),
            settings.llm_model,
            settings.llm_base_url,
        )
        yield

    app = FastAPI(
        title="Geometry Assistant",
        version="0.1.0",
        description="Natural-language questions about 3D models, answered by whitelisted tools.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    window = settings.ask_rate_window_s
    client_limiter = RateLimiter(settings.ask_rate_limit, window)
    global_limiter = RateLimiter(settings.ask_global_limit, window)
    uploads = SessionStore(settings.max_sessions, settings.max_uploads_per_session)

    def samples(request: Request) -> Catalog:
        return request.app.state.samples

    def cat(request: Request) -> Catalog:
        """Models visible to a request: the samples and the uploads of its session."""
        return samples(request).merged(uploads.catalog(_session_id(request)))

    def summary(request: Request, model_id: str, catalog: Catalog) -> dict[str, Any]:
        source = "sample" if model_id in samples(request).models else "upload"
        return {**describe_model(model_id, catalog.models[model_id]), "source": source}

    def check_rate_limit(request: Request) -> None:
        """Count one question; raise 429 with a friendly message when a limit is reached."""
        ip = _client_ip(request, settings.trust_forwarded_for)
        blocked: tuple[float, str] | None = None
        if settings.ask_rate_limit and (wait := client_limiter.retry_after(ip)) > 0:
            blocked = (
                wait,
                f"You have reached the limit of {settings.ask_rate_limit} questions per "
                f"{_window_text(window)} for this public demo.",
            )
        elif settings.ask_global_limit and (wait := global_limiter.retry_after(GLOBAL_KEY)) > 0:
            blocked = (wait, "This public demo has answered its quota of questions for now.")
        if blocked:
            wait, message = blocked
            logger.info("Rate limit reached for %s (retry in %.0f s)", ip, wait)
            raise HTTPException(
                429,
                f"{message} Please try again in about {_wait_text(wait)}. "
                "The tools remain available without limit through POST /tools.",
                headers={"Retry-After": str(math.ceil(wait))},
            )
        if settings.ask_rate_limit:
            client_limiter.hit(ip)
        if settings.ask_global_limit:
            global_limiter.hit(GLOBAL_KEY)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """3D viewer and chat (keeps the conversation in the browser)."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Liveness probe: sample models and configured LLM (no LLM call)."""
        return {
            "status": "ok",
            "models": sorted(samples(request).models),
            "llm_model": settings.llm_model,
            "llm_configured": llm_configured,
        }

    @app.get("/models", response_model=list[ModelSummary])
    def list_models(request: Request) -> list[dict[str, Any]]:
        """Sample models, then the models uploaded in this session."""
        catalog = cat(request)
        return [summary(request, model_id, catalog) for model_id in catalog.models]

    @app.get(
        "/models/{model_id}/mesh",
        response_class=Response,
        responses={200: {"content": {"model/stl": {}}}},
    )
    def model_mesh(request: Request, model_id: str) -> Response:
        """Mesh of a model as binary STL.

        The viewer loads this file rather than the uploaded one, so triangle i
        on screen is face i of the tools.
        """
        catalog = cat(request)
        if model_id not in catalog.models:
            raise HTTPException(404, f"Unknown model '{model_id}'.")
        return Response(to_stl(catalog.models[model_id]), media_type="model/stl")

    @app.post("/models", status_code=201, response_model=ModelSummary, openapi_extra=UPLOAD_BODY)
    async def upload_model(request: Request, response: Response) -> dict[str, Any]:
        """Upload an STL or OBJ file (multipart field `file`); kept in memory for the session."""
        limit = settings.max_upload_bytes
        too_large = HTTPException(413, f"The file is larger than {settings.max_upload_mb:g} MB.")
        # Refuse oversized bodies before reading them.
        declared = request.headers.get("content-length", "")
        if not declared.isdigit():
            raise HTTPException(411, "A Content-Length header is required.")
        if int(declared) > limit + FORM_OVERHEAD_BYTES:
            raise too_large

        async with request.form(max_files=1, max_fields=1) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "Send the model as the multipart field 'file'.")
            name = Path(upload.filename or "model")
            file_type = name.suffix.lower().lstrip(".")
            if file_type not in MESH_FILE_TYPES:
                raise HTTPException(415, "Unsupported file type: upload an .stl or .obj file.")
            data = await upload.read(limit + 1)
        if len(data) > limit:
            raise too_large

        try:
            mesh = await run_in_threadpool(load_mesh, data, file_type, settings.max_faces)
        except MeshError as exc:
            raise HTTPException(422, str(exc)) from exc

        session = _session_id(request) or secrets.token_urlsafe(24)
        reserved = frozenset(samples(request).models)
        model_id = uploads.add(session, name.stem, mesh, reserved=reserved)
        response.set_cookie(
            SESSION_COOKIE, session, max_age=SESSION_MAX_AGE_S, httponly=True, samesite="lax"
        )
        return {**describe_model(model_id, mesh), "source": "upload"}

    @app.post("/tools")
    def call_tool(request: Request, body: ToolCall) -> dict[str, Any]:
        """Call a geometry tool directly (useful for testing without an LLM)."""
        if body.tool not in TOOLS:
            raise HTTPException(404, f"Unknown tool '{body.tool}'. Available: {sorted(TOOLS)}")
        result = run_tool(cat(request), body.tool, body.args)
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result

    @app.post("/ask", response_model=Answer)
    def ask_question(request: Request, body: Question) -> dict[str, Any]:
        """Natural-language question -> LLM -> geometry tools -> answer, trace and view actions."""
        if not llm_configured:
            raise HTTPException(
                503,
                "No language model is configured on this server (LLM_API_KEY is not set). "
                "The tools remain available through POST /tools.",
            )
        check_rate_limit(request)
        try:
            return ask(
                body.question,
                cat(request),
                selection=body.selection,
                history=[t.model_dump() for t in body.history],
                settings=settings,
            )
        except RateLimitError as exc:
            # Provider messages can name the account, so they are logged, not returned.
            logger.error("LLM quota exhausted: %s", exc)
            raise HTTPException(
                503,
                "The language model is receiving too many requests right now. "
                "Please try again in a few minutes.",
            ) from exc
        except OpenAIError as exc:
            logger.error("LLM request failed: %s", exc)
            raise HTTPException(
                502, "The language model could not be reached. Please try again later."
            ) from exc

    return app


app = create_app()
