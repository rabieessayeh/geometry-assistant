"""Scoring of the agent's tool choices against an evaluation set.

The evaluation does not judge the wording of the answers. Because every number
comes from a deterministic tool, an answer is right when the LLM picked the
right tool with the right arguments, and that is what is measured here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Reading the description of a model before acting on it is always allowed.
DISCOVERY_TOOL = "model_info"


@dataclass(frozen=True)
class Case:
    """One evaluation question and the tool call a correct agent should make."""

    id: str
    question: str
    tool: str | None  # None: no tool can answer, so none should be called
    args: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)  # viewer state sent with it


@dataclass(frozen=True)
class Score:
    """Outcome of one case."""

    case: Case
    called: list[str]
    tool_ok: bool
    args_ok: bool


def load_cases(path: str | Path) -> list[Case]:
    """Read the evaluation set from a YAML file."""
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cases = [
        Case(
            id=entry["id"],
            question=entry["question"],
            tool=entry["expected"]["tool"],
            args=entry["expected"].get("args") or {},
            selection=entry.get("selection") or {},
        )
        for entry in entries
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case ids in {path}")
    return cases


def _same(expected: Any, actual: Any) -> bool:
    """Compare one argument; numbers match within a small tolerance, also inside lists."""
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(e, a) for e, a in zip(expected, actual, strict=True))
        )
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return expected == actual
    try:
        return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True if every expected argument is present with the expected value."""
    return all(key in actual and _same(value, actual[key]) for key, value in expected.items())


def score_case(case: Case, trace: list[dict[str, Any]]) -> Score:
    """Score the trace of tool calls produced for one case.

    The tool is right when the expected tool was called and no other tool was
    (apart from `model_info`). The arguments are right when at least one call
    of that tool carries the expected arguments.
    """
    called = [step["tool"] for step in trace]
    if case.tool is None:
        ok = all(tool == DISCOVERY_TOOL for tool in called)
        return Score(case, called, tool_ok=ok, args_ok=ok)
    tool_ok = case.tool in called and all(t in {case.tool, DISCOVERY_TOOL} for t in called)
    args_ok = tool_ok and any(
        step["tool"] == case.tool and args_match(case.args, step["args"]) for step in trace
    )
    return Score(case, called, tool_ok=tool_ok, args_ok=args_ok)


def summarise(scores: list[Score]) -> dict[str, Any]:
    """Aggregate scores into accuracies (fractions between 0 and 1)."""
    n = len(scores)
    return {
        "cases": n,
        "tool_accuracy": sum(s.tool_ok for s in scores) / n if n else 0.0,
        "args_accuracy": sum(s.args_ok for s in scores) / n if n else 0.0,
    }


def format_report(scores: list[Score]) -> str:
    """Render a plain-text table of the results."""
    width = max((len(s.case.id) for s in scores), default=4)
    lines = [f"{'case':<{width}}  tool  args  called"]
    for s in scores:
        marks = f"{'ok' if s.tool_ok else 'FAIL':<4}  {'ok' if s.args_ok else 'FAIL':<4}"
        lines.append(f"{s.case.id:<{width}}  {marks}  {', '.join(s.called) or '-'}")
    summary = summarise(scores)
    lines.append("")
    lines.append(f"Tool accuracy:      {summary['tool_accuracy']:.0%} of {summary['cases']} cases")
    lines.append(f"Argument accuracy:  {summary['args_accuracy']:.0%} of {summary['cases']} cases")
    return "\n".join(lines) + "\n"
