"""Evaluation scoring, and consistency of eval/questions.yaml with the tools and samples."""

from pathlib import Path

import pytest

from app.agent import Selection
from app.evaluation import Case, args_match, format_report, load_cases, score_case, summarise
from app.geometry import TOOLS, Catalog, run_tool

ROOT = Path(__file__).resolve().parents[1]
CASES = load_cases(ROOT / "eval" / "questions.yaml")

SLICE = Case("c", "q", "slice_at", {"model": "bridge", "axis": "z", "value_mm": 10})


@pytest.fixture(scope="module")
def samples() -> Catalog:
    """The sample files shipped in data/samples."""
    return Catalog.from_folder(ROOT / "data" / "samples")


def step(tool: str, **args) -> dict:
    return {"tool": tool, "args": args, "ok": True, "summary": ""}


def test_args_match_tolerates_number_formats_and_extra_arguments():
    assert args_match({"value_mm": 10}, {"value_mm": 10.0, "extra": 1})
    assert args_match({"value_mm": 10}, {"value_mm": "10"})
    assert not args_match({"value_mm": 10}, {"value_mm": 1})
    assert not args_match({"value_mm": 10}, {"value_mm": "high"})
    assert not args_match({"axis": "z"}, {})
    assert not args_match({"axis": "z"}, {"axis": "x"})


def test_args_match_compares_points():
    assert args_match({"point_a": [0, 0, 25]}, {"point_a": [0.0, 0.0, 25.0]})
    assert not args_match({"point_a": [0, 0, 25]}, {"point_a": [0, 0, 24]})
    assert not args_match({"point_a": [0, 0, 25]}, {"point_a": [0, 0]})
    assert not args_match({"point_a": [0, 0, 25]}, {"point_a": "P1"})


def test_correct_call_after_discovery():
    trace = [
        step("model_info", model="bridge"),
        step("slice_at", model="bridge", axis="z", value_mm=10),
    ]
    score = score_case(SLICE, trace)
    assert (score.tool_ok, score.args_ok) == (True, True)


def test_right_tool_wrong_arguments():
    trace = [step("slice_at", model="bridge", axis="x", value_mm=10)]
    score = score_case(SLICE, trace)
    assert (score.tool_ok, score.args_ok) == (True, False)


def test_retry_with_corrected_arguments_counts():
    trace = [
        step("slice_at", model="bridge", axis="z", value_mm=100),
        step("slice_at", model="bridge", axis="z", value_mm=10),
    ]
    assert score_case(SLICE, trace).args_ok


@pytest.mark.parametrize(
    "trace",
    [
        [],
        [step("overhang_analysis", model="bridge")],
        [step("slice_at", model="bridge", axis="z", value_mm=10), step("set_view", view="top")],
    ],
)
def test_wrong_missing_or_extra_tool_fails(trace):
    score = score_case(SLICE, trace)
    assert (score.tool_ok, score.args_ok) == (False, False)


def test_out_of_scope_case():
    case = Case("c", "q", None)
    assert score_case(case, []).tool_ok
    assert score_case(case, [step("model_info", model="bridge")]).tool_ok
    assert not score_case(case, [step("set_view", view="top")]).tool_ok


def test_summary_and_report():
    scores = [
        score_case(SLICE, [step("slice_at", model="bridge", axis="z", value_mm=10)]),
        score_case(SLICE, [step("slice_at", model="bridge", axis="y", value_mm=10)]),
    ]
    assert summarise(scores) == {"cases": 2, "tool_accuracy": 1.0, "args_accuracy": 0.5}
    report = format_report(scores)
    assert "Tool accuracy:      100% of 2 cases" in report
    assert "Argument accuracy:  50% of 2 cases" in report


def test_duplicate_ids_are_rejected(tmp_path):
    entry = "- {id: a, question: q, expected: {tool: null}}\n"
    path = tmp_path / "questions.yaml"
    path.write_text(entry * 2)
    with pytest.raises(ValueError, match="Duplicate case ids"):
        load_cases(path)


def test_question_file_size():
    assert 8 <= len(CASES) <= 10


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_cases_are_consistent_with_the_tools_and_samples(case, samples):
    """Every selection is valid, and every expected call runs on the sample models."""
    selection = Selection(**case.selection)
    assert selection.model in samples.models
    if case.tool is None:
        return
    assert case.tool in TOOLS
    result = run_tool(samples, case.tool, case.args)
    assert "error" not in result
