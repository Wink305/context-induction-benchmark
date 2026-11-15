"""Command-line entry points. All commands are offline unless explicitly enabled."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adjudication import freeze_gold
from .agreement import agreement_report
from .audit import audit_dataset
from .generator import generate_dataset
from .harness import InductionRunner, write_call_logs
from .io import read_json, read_jsonl, write_json, write_jsonl
from .providers import MockProvider, OpenAICompatibleProvider
from .scoring import judge_calibration, score_counterfactual, score_dataset
from .sheets import make_sheets, parse_sheets
from .survival import survival_report
from .validation import public_constraint, validate_candidate_dialogue


def _provider(config: dict[str, Any]):
    if config["provider"] == "mock":
        return MockProvider()
    if config["provider"] == "openai_compatible":
        return OpenAICompatibleProvider(
            config["base_url"], api_key_env=config["api_key_env"],
            allow_api_calls=config.get("allow_api_calls", False),
            input_price_per_million=config.get("input_price_per_million"),
            output_price_per_million=config.get("output_price_per_million"),
        )
    raise ValueError(f"Unknown provider: {config['provider']}")


def _self_check(candidate_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    dialogues = read_jsonl(candidate_path)
    validation = [validate_candidate_dialogue(dialogue) for dialogue in dialogues]
    predictions = [
        {"dialogue_id": dialogue["dialogue_id"], "output": {"constraints": [public_constraint(item) for item in dialogue["candidate_truth"]]}}
        for dialogue in dialogues
    ]
    scored = score_dataset(dialogues, predictions)
    result = {
        "scientific_status": "validator_and_scorer_self_check_not_a_model_result",
        "candidate_validation_pass_n": sum(report.valid for report in validation),
        "candidate_dialogue_n": len(dialogues),
        "candidate_issues": [report.to_dict() for report in validation if not report.valid],
        "oracle_aggregate": scored["aggregate"],
    }
    if output_path:
        write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate unannotated P0 candidate dialogues")
    generate.add_argument("--config", default="configs/generator.json")

    validate = sub.add_parser("validate", help="Validate candidate dialogues")
    validate.add_argument("--candidate", default="data/pilot_v0/candidate_truth.jsonl")
    validate.add_argument("--output")

    check = sub.add_parser("self-check", help="Run oracle-input validator/scorer checks; not a model evaluation")
    check.add_argument("--candidate", default="data/pilot_v0/candidate_truth.jsonl")
    check.add_argument("--output")

    score = sub.add_parser("score", help="Score induction predictions")
    score.add_argument("--truth", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)

    score_b = sub.add_parser("score-b", help="Score true/none/fake counterfactual results")
    score_b.add_argument("--input", required=True)
    score_b.add_argument("--output", required=True)

    agree = sub.add_parser("agreement", help="Compute P0 annotation agreement and adjudication queue")
    agree.add_argument("--a", required=True)
    agree.add_argument("--b", required=True)
    agree.add_argument("--output", required=True)
    agree.add_argument("--bootstrap", type=int, default=1000)

    judge = sub.add_parser("judge-calibration", help="Report Gold:Yes/Gold:No errors and unextractable rate")
    judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True)

    freeze = sub.add_parser("freeze-gold", help="Freeze human-adjudicated annotations; never auto-resolves")
    freeze.add_argument("--candidate", required=True)
    freeze.add_argument("--annotations", required=True)
    freeze.add_argument("--agreement-report", required=True)
    freeze.add_argument("--resolutions", required=True, help="JSON list of explicit third-reviewer decisions")
    freeze.add_argument("--output-dir", required=True)

    survival = sub.add_parser("survival", help="Compute R1 Kaplan-Meier and recovery metrics")
    survival.add_argument("--input", required=True)
    survival.add_argument("--output", required=True)
    survival.add_argument("--bootstrap", type=int, default=1000)

    audit = sub.add_parser("audit", help="Rule-audit candidate truth: recoverability, type cues, bookkeeping")
    audit.add_argument("--candidate", default="data/pilot_v0/candidate_truth.jsonl")
    audit.add_argument("--output", required=True)

    sheets = sub.add_parser("make-sheets", help="Render human-fillable annotation sheets into Markdown")
    sheets.add_argument("--worksheet", default="data/pilot_v0/annotation_worksheet.jsonl")
    sheets.add_argument("--out-dir", default="data/annotation/sheets_A")
    sheets.add_argument("--calibration", default="data/pilot_v0/calibration_ids.json")
    sheets.add_argument("--annotator", default="A")
    sheets.add_argument("--ids", nargs="*")

    parse = sub.add_parser("parse-sheets", help="Parse filled annotation sheets into annotation records")
    parse.add_argument("--dir", required=True)
    parse.add_argument("--annotator", default="A")
    parse.add_argument("--output", required=True)
    parse.add_argument("--report", required=True)

    run = sub.add_parser("run-induction", help="Run the two-stage harness induction stage")
    run.add_argument("--runtime", default="configs/runtime.json")
    run.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    run.add_argument("--prompt", default="prompts/induction_v1.txt")
    run.add_argument("--output", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_dataset(args.config)
    elif args.command == "validate":
        dialogues = read_jsonl(args.candidate)
        reports = [{"dialogue_id": dialogue["dialogue_id"], **validate_candidate_dialogue(dialogue).to_dict()} for dialogue in dialogues]
        result = {"dialogue_n": len(reports), "valid_n": sum(row["valid"] for row in reports), "reports": reports}
        if args.output:
            write_json(args.output, result)
    elif args.command == "self-check":
        result = _self_check(Path(args.candidate), Path(args.output) if args.output else None)
    elif args.command == "score":
        result = score_dataset(read_jsonl(args.truth), read_jsonl(args.predictions))
        write_json(args.output, result)
    elif args.command == "score-b":
        result = score_counterfactual(read_jsonl(args.input))
        write_json(args.output, result)
    elif args.command == "agreement":
        result = agreement_report(read_jsonl(args.a), read_jsonl(args.b), bootstrap_n=args.bootstrap)
        write_json(args.output, result)
    elif args.command == "judge-calibration":
        result = judge_calibration(read_jsonl(args.input))
        write_json(args.output, result)
    elif args.command == "freeze-gold":
        result = freeze_gold(
            read_jsonl(args.candidate), read_jsonl(args.annotations), read_json(args.agreement_report),
            read_json(args.resolutions), args.output_dir,
        )
    elif args.command == "survival":
        result = survival_report(read_json(args.input), bootstrap_n=args.bootstrap)
        write_json(args.output, result)
    elif args.command == "audit":
        result = audit_dataset(read_jsonl(args.candidate))
        write_json(args.output, result)
    elif args.command == "make-sheets":
        calibration = read_json(args.calibration)["dialogue_ids"] if args.calibration and Path(args.calibration).exists() else []
        result = make_sheets(
            args.worksheet,
            args.out_dir,
            ids=args.ids,
            annotator=args.annotator,
            calibration_ids=calibration,
        )
    elif args.command == "parse-sheets":
        parsed = parse_sheets(Path(args.dir), annotator=args.annotator)
        annotations = parsed.pop("annotations")
        write_jsonl(args.output, annotations)
        write_json(args.report, parsed)
        result = {key: value for key, value in parsed.items() if key != "questions"} | {
            "questions_n": len(parsed["questions"]),
        }
    elif args.command == "run-induction":
        config = read_json(args.runtime)
        if not config.get("model_id") or not config.get("pinned_version"):
            raise ValueError("A verified model_id and pinned_version are required")
        tasks = read_jsonl(args.tasks)
        if args.limit:
            tasks = tasks[:args.limit]
        runner = InductionRunner(_provider(config), config, Path(args.prompt).read_text(encoding="utf-8"))
        results = [runner.run(task) for task in tasks]
        write_jsonl(args.output, results)
        write_call_logs(args.logs, results)
        result = {"run_n": len(results), "valid_n": sum(row["valid"] for row in results), "failed_n": sum(not row["valid"] for row in results), "status": "engineering_run_not_a_benchmark_result"}
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
