"""Measure how often the configured LLM picks the right tool and arguments.

Each question of `eval/questions.yaml` is sent to the agent with its selection
context; the trace of tool calls is compared with the expected call. Needs a
reachable LLM (see `.env`).

Usage:  python eval/run_eval.py [--models data/samples] [--delay 2] [--out eval/results/run.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from openai import OpenAIError

from app.agent import Selection, ask
from app.config import configure_logging, get_settings
from app.evaluation import Score, format_report, load_cases, score_case, summarise
from app.geometry import Catalog

logger = logging.getLogger("run_eval")

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the evaluation and print a report; exit code 1 if a case could not be run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "data" / "samples", help="folder")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--out", type=Path, help="write detailed results to this JSON file")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")
    catalog = Catalog.from_folder(args.models)
    cases = load_cases(args.questions)

    scores: list[Score] = []
    errors: dict[str, str] = {}
    for i, case in enumerate(cases):
        if i and args.delay:
            time.sleep(args.delay)
        try:
            selection = Selection(**case.selection)
            result = ask(case.question, catalog, selection=selection, settings=settings)
        except OpenAIError as exc:
            logger.error("Case '%s' could not be run: %s", case.id, exc)
            errors[case.id] = str(exc)
            continue
        scores.append(score_case(case, result["trace"]))

    sys.stdout.write(f"Model: {settings.llm_model} ({settings.llm_base_url})\n\n")
    sys.stdout.write(format_report(scores))
    if errors:
        sys.stdout.write(f"\n{len(errors)} case(s) not run because of LLM errors.\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "summary": summarise(scores),
            "cases": [asdict(s) for s in scores],
            "errors": errors,
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
