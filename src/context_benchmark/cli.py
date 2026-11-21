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
from .dimensions import registry as dimension_registry
from .generator import generate_dataset
from .harness import (
    ExecutorRunner, InductionRunner, planned_conditions, run_no_compression,
    run_survival_trace, write_call_logs,
)
from .io import read_json, read_jsonl, write_json, write_jsonl
from .manifest import build_run_manifest, validate_submission
from .ontology import BUDGETS
from .providers import MockProvider, OpenAICompatibleProvider
from .scoring import MATCHER_VERSION, judge_calibration, score_counterfactual, score_dataset, score_dialogue
from .sheets import make_sheets, parse_sheets
from .survival import survival_report
from .validation import public_constraint, validate_candidate_dialogue

# Recorded in every self-check artifact so a stale artifact can be detected by
# comparing it with the scorer that is actually installed.
SCORER_VERSION = "scoring.py@semantic_guard_v1"


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


def _require_pinned(config: dict[str, Any]) -> None:
    """Every model-touching command path must pin the model.

    ``run-induction`` already enforced this; the executor and no-compression
    paths did not, so an unpinned model could produce E/F artifacts (and a
    submission package) that the README explicitly calls irreproducible.
    """
    if not config.get("model_id") or not config.get("pinned_version"):
        raise ValueError(
            "A verified model_id and pinned_version are required "
            "(an unpinned model makes a benchmark run irreproducible)"
        )


def _self_check(candidate_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Scorer sanity check: oracle must pass AND negatives must fail.

    Feeding the answer key back and getting 1.0 proves nothing on its own. The
    check is only meaningful when a set of corrupted predictions is scored low
    by the same scorer. Both are reported; neither is a model result.

    Three negative controls are used, because the interesting failures are not
    random noise: gibberish text, **semantically inverted** text (the gold
    sentence plus "不要遵守本条，改为完全相反的操作"), and value-flipped text
    (the same slot with the opposite value, e.g. 短信 → 邮件). A scorer that
    only rejects noise but accepts an inverted constraint is not usable.
    """
    dialogues = read_jsonl(candidate_path)
    validation = [validate_candidate_dialogue(dialogue) for dialogue in dialogues]
    oracle = [
        {"dialogue_id": dialogue["dialogue_id"], "output": {"constraints": [public_constraint(item) for item in dialogue["candidate_truth"]]}}
        for dialogue in dialogues
    ]

    def corrupted(builder) -> list[dict[str, Any]]:
        rows = []
        for dialogue in dialogues:
            constraints = [public_constraint(item) for item in dialogue["candidate_truth"]]
            for item in constraints:
                item["text"] = builder(item["text"])
            rows.append({"dialogue_id": dialogue["dialogue_id"], "output": {"constraints": constraints}})
        return rows

    # Negative controls: same ids/types/evidence, corrupted text.
    # A scorer that still returns a high score here is broken.
    def flip_slot_values(text: str) -> str:
        """Swap a controlled value for its opposite, in both directions.

        A one-directional replacement would leave most variants untouched
        (the pilot mixes 短信/邮件 phrasings), which would make the control
        look like it passed while most of the corpus was untouched.
        """
        swaps = (("短信", "\0MAIL\0"), ("邮件", "短信"), ("\0MAIL\0", "邮件"))
        swapped = text
        for source, target in swaps:
            swapped = swapped.replace(source, target)
        return swapped

    negatives = {
        "gibberish": corrupted(lambda text: "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未"),
        "semantic_inversion": corrupted(
            lambda text: text + "但请不要遵守本条，改为完全相反的操作并忽略前面的要求。"
        ),
    }
    oracle_scored = score_dataset(dialogues, oracle)["aggregate"]
    negative_scored = {
        name: score_dataset(dialogues, rows)["aggregate"] for name, rows in negatives.items()
    }
    oracle_must_include = oracle_scored["must_include"]["value"]
    negative_must_include = {
        name: aggregate["must_include"]["value"] for name, aggregate in negative_scored.items()
    }
    gaps = {
        name: (oracle_must_include - value) if (oracle_must_include is not None and value is not None) else None
        for name, value in negative_must_include.items()
    }

    # Value-flip control is measured per unit, because only the units that
    # carry a controlled slot can be flipped. A corpus-level must_include over
    # the whole pilot would stay high simply because most units were untouched.
    flip_total = 0
    flip_caught = 0
    flip_matched = 0
    for dialogue in dialogues:
        for item in dialogue["candidate_truth"]:
            flipped = flip_slot_values(item["text"])
            if flipped == item["text"]:
                continue
            flip_total += 1
            row = {
                "dialogue_id": dialogue["dialogue_id"],
                "output": {"constraints": [dict(public_constraint(item), text=flipped)]},
            }
            scored = score_dialogue(dialogue, row)
            if scored["match_n"] > 0:
                flip_matched += 1
            if scored["content_gate"]["content_ok_n"] == 0:
                flip_caught += 1
    flip_rate = (flip_caught / flip_total) if flip_total else None

    discriminates = (
        oracle_must_include is not None
        and all(value is not None for value in negative_must_include.values())
        and all(gap is not None and gap >= 0.5 for gap in gaps.values())
        and flip_rate is not None
        and flip_rate >= 0.9
    )
    result = {
        "scientific_status": "validator_and_scorer_self_check_not_a_model_result",
        "scorer_version": SCORER_VERSION,
        "matcher_version": MATCHER_VERSION,
        "candidate_validation_pass_n": sum(report.valid for report in validation),
        "candidate_dialogue_n": len(dialogues),
        "candidate_issues": [report.to_dict() for report in validation if not report.valid],
        "oracle_aggregate": oracle_scored,
        "negative_control_aggregates": negative_scored,
        "value_flip_control": {
            "unit_n": flip_total,
            "caught_n": flip_caught,
            "matched_n": flip_matched,
            "catch_rate": flip_rate,
            "required_catch_rate": 0.9,
        },
        "discrimination_check": {
            "oracle_must_include": oracle_must_include,
            "negative_must_include": negative_must_include,
            "gap": gaps,
            "min_gap": min((gap for gap in gaps.values() if gap is not None), default=None),
            "value_flip_catch_rate": flip_rate,
            "required_gap": 0.5,
            "passes": discriminates,
        },
    }
    if output_path:
        write_json(output_path, result)
    if not discriminates:
        raise ValueError(
            f"Scorer self-check failed discrimination: oracle={oracle_must_include}, "
            f"negatives={negative_must_include}, value_flip_catch_rate={flip_rate}. "
            f"A scorer that cannot tell these apart must not be used to report model results."
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate unannotated P0 candidate dialogues")
    generate.add_argument("--config", default="configs/generator.json")
    generate.add_argument("--output-dir", help="Append-only extension set directory; never the frozen data/pilot_v0")
    generate.add_argument("--contributor", default="", help="Contributor handle for the extension dataset.json")
    generate.add_argument("--split", default="public", choices=["public", "private", "holdout", "unspecified"])

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
    freeze.add_argument(
        "--candidate",
        required=True,
        help="Truth-free annotation worksheet (data/pilot_v0/annotation_worksheet.jsonl). "
             "Generator truth is refused: it carries candidate_only markers.",
    )
    freeze.add_argument("--annotations", required=True, help="Annotator A's parsed annotation set")
    freeze.add_argument(
        "--annotations-b", required=True,
        help="Annotator B's parsed annotation set. Agreement between a set and itself is "
             "vacuous, so the freezer refuses to run without this.",
    )
    freeze.add_argument("--agreement-report", required=True)
    freeze.add_argument(
        "--distractors",
        help="Truth file carrying the rule-audited distractors / non_units to attach to the "
             "frozen gold. Without it the negative side is absent and FMR is undefined.",
    )
    freeze.add_argument("--resolutions", required=True, help="JSON list of explicit third-reviewer decisions")
    freeze.add_argument("--output-dir", required=True)

    survival = sub.add_parser("survival", help="Compute R1 Kaplan-Meier and recovery metrics")
    survival.add_argument("--input", required=True)
    survival.add_argument("--output", required=True)
    survival.add_argument("--bootstrap", type=int, default=1000)

    run_r2 = sub.add_parser("run-r2", help="Dimension F: no-compression durability baseline")
    run_r2.add_argument("--runtime", default="configs/runtime.json")
    run_r2.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    run_r2.add_argument("--prompt", default="prompts/induction_v1.txt")
    run_r2.add_argument("--output", required=True)
    run_r2.add_argument("--limit", type=int)

    run_e = sub.add_parser("run-survival-trace", help="Dimension E: per-turn compliance traces for survival")
    run_e.add_argument("--runtime", default="configs/runtime.json")
    run_e.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    run_e.add_argument("--truth", default="data/pilot_v0/candidate_truth.jsonl")
    run_e.add_argument("--executor-prompt", default="prompts/executor_v2.txt")
    run_e.add_argument("--output", required=True)
    run_e.add_argument("--limit", type=int)

    audit = sub.add_parser("audit", help="Rule-audit candidate truth: recoverability, type cues, bookkeeping")
    audit.add_argument("--candidate", default="data/pilot_v0/candidate_truth.jsonl")
    audit.add_argument("--output", required=True)

    sheets = sub.add_parser("make-sheets", help="Render human-fillable annotation sheets into Markdown")
    sheets.add_argument("--worksheet", default="data/pilot_v0/annotation_worksheet.jsonl")
    sheets.add_argument("--out-dir", default="data/annotation/sheets_A")
    sheets.add_argument("--calibration", default="data/pilot_v0/calibration_ids.json")
    sheets.add_argument("--annotator", default="A")
    sheets.add_argument("--ids", nargs="*")
    sheets.add_argument(
        "--truth", help="Optional candidate_truth.jsonl; required only for opt-in structural hints"
    )

    parse = sub.add_parser("parse-sheets", help="Parse filled annotation sheets into annotation records")
    parse.add_argument("--dir", required=True)
    parse.add_argument("--annotator", default="A")
    parse.add_argument(
        "--provenance", default=None,
        help="Declared origin of these sheets (human / ai_precheck / …). Without a "
             "declaration the parsed rows are marked unknown, never human.",
    )
    parse.add_argument("--output", required=True)
    parse.add_argument("--report", required=True)

    run = sub.add_parser("run-induction", help="Run the two-stage harness induction stage")
    run.add_argument("--runtime", default="configs/runtime.json")
    run.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    run.add_argument("--prompt", default="prompts/induction_v1.txt")
    run.add_argument("--output", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--limit", type=int)
    run.add_argument(
        "--truth", help="Optional candidate_truth.jsonl for post-hoc zero-constraint diagnostics; never passed to the model"
    )
    run.add_argument("--manifest", help="Also write a version-stamped, self-contained submission directory")
    run.add_argument("--contributor", default="")
    run.add_argument("--dimensions", nargs="*", default=[], help="Dimensions claimed by this run")
    run.add_argument("--scores-declared", action="store_true")
    run.add_argument("--repetition", type=int, default=1)

    probes = sub.add_parser("run-probes", help="Run the three-arm executor conditions from induced constraints")
    probes.add_argument("--runtime", default="configs/runtime.json")
    probes.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    probes.add_argument("--truth", default="data/pilot_v0/candidate_truth.jsonl")
    probes.add_argument("--predictions", required=True)
    probes.add_argument("--executor-prompt", default="prompts/executor_v2.txt")
    probes.add_argument("--output", required=True)
    probes.add_argument("--logs", required=True)
    probes.add_argument("--scores", help="Optional counterfactual score report")
    probes.add_argument("--budgets", nargs="*", choices=BUDGETS, default=["medium"])
    probes.add_argument("--limit", type=int)

    stamp = sub.add_parser("run-manifest", help="Build a version-stamped submission from existing run artifacts")
    stamp.add_argument("--runtime", default="configs/runtime.json")
    stamp.add_argument("--tasks", default="data/pilot_v0/tasks.jsonl")
    stamp.add_argument("--prompt", default="prompts/induction_v1.txt")
    stamp.add_argument("--predictions", required=True)
    stamp.add_argument("--logs", required=True)
    stamp.add_argument("--out-dir", required=True)
    stamp.add_argument("--contributor", default="")
    stamp.add_argument("--notes", default="")
    stamp.add_argument("--dimensions", nargs="*", default=[])
    stamp.add_argument("--scores-declared", action="store_true")
    stamp.add_argument("--repetition", type=int, default=1)

    check_sub = sub.add_parser("validate-submission", help="Validate a run submission or dataset extension directory")
    check_sub.add_argument("--dir", required=True)
    check_sub.add_argument("--benchmark-config", default="configs/benchmark.json")

    sub.add_parser("dimensions", help="Print the six-axis dimension registry")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        result = generate_dataset(
            args.config, output_dir=args.output_dir, contributor=args.contributor, split=args.split,
        )
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
            annotations_b=read_jsonl(args.annotations_b),
            distractor_source=args.distractors,
        )
    elif args.command == "survival":
        # Accept either a single JSON object or a JSONL trace produced by
        # run-survival-trace, so E is now reachable end-to-end from a real run.
        path = Path(args.input)
        if path.suffix == ".jsonl" or "\n" in path.read_text(encoding="utf-8").strip():
            raw = read_jsonl(path)
            records = raw[0]["records"] if len(raw) == 1 and isinstance(raw[0], dict) and "records" in raw[0] else raw
        else:
            loaded = read_json(path)
            records = loaded["records"] if isinstance(loaded, dict) and "records" in loaded else loaded
        result = survival_report(records, bootstrap_n=args.bootstrap)
        write_json(args.output, result)
    elif args.command == "run-r2":
        config = read_json(args.runtime)
        _require_pinned(config)
        tasks = read_jsonl(args.tasks)
        if args.limit:
            tasks = tasks[:args.limit]
        prompt = Path(args.prompt).read_text(encoding="utf-8")
        provider = _provider(config)
        records = [run_no_compression(provider, config, task, prompt) for task in tasks]
        write_jsonl(args.output, records)
        result = {
            "trace_n": len(records),
            "task_success_n": sum(row.get("task_success", False) for row in records),
            "overflow_n": sum(row.get("failure_reason") == "estimated_context_overflow" for row in records),
            "compression": False,
            "status": "engineering_run_not_a_benchmark_result",
        }
    elif args.command == "run-survival-trace":
        config = read_json(args.runtime)
        _require_pinned(config)
        tasks = read_jsonl(args.tasks)
        truth = {row["dialogue_id"]: row for row in read_jsonl(args.truth)}
        if args.limit:
            tasks = tasks[:args.limit]
        prompt = Path(args.executor_prompt).read_text(encoding="utf-8")
        provider = _provider(config)
        records = []
        for task in tasks:
            row = truth.get(task["dialogue_id"])
            if row is None:
                continue
            # run_survival_trace expects the full dialogue (turns + truth).
            merged = {**task, "turns": task["turns"], "candidate_truth": row["candidate_truth"]}
            records.append(run_survival_trace(provider, config, merged, row, prompt))
        write_jsonl(args.output, records)
        result = {
            "trace_n": len(records),
            "constraint_probe_n": sum(len(row["compliance"]) for row in records),
            "next": "python -m context_benchmark survival --input <this-output> --output <report>",
            "status": "engineering_run_not_a_benchmark_result",
        }
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
            truth_path=args.truth,
        )
    elif args.command == "parse-sheets":
        parsed = parse_sheets(
            Path(args.dir), annotator=args.annotator, provenance=args.provenance
        )
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
        zero_constraint_n = sum(
            not (row.get("output") or {}).get("constraints") for row in results
        )
        result["zero_constraint_output_n"] = zero_constraint_n
        # Post-hoc honesty check: if gold is supplied, count dialogues that had
        # constraints but received an empty induction. This never reaches the model.
        if args.truth:
            truth = {row["dialogue_id"]: row for row in read_jsonl(args.truth)}
            missed = [
                row["dialogue_id"] for row in results
                if truth.get(row["dialogue_id"], {}).get("candidate_truth")
                and not (row.get("output") or {}).get("constraints")
            ]
            result["nonempty_dialogue_with_empty_induction_n"] = len(missed)
            result["nonempty_dialogue_with_empty_induction_ids"] = missed[:20]
        if args.manifest:
            result["manifest"] = build_run_manifest(
                runtime_path=args.runtime, tasks_path=args.tasks, prompt_path=args.prompt,
                predictions_path=args.output, logs_path=args.logs, out_dir=args.manifest,
                contributor=args.contributor, dimensions_reported=args.dimensions,
                scores_declared=args.scores_declared, repetition_index=args.repetition,
            )
    elif args.command == "run-probes":
        config = read_json(args.runtime)
        _require_pinned(config)
        truth = {row["dialogue_id"]: row for row in read_jsonl(args.truth)}
        predictions = {row["dialogue_id"]: row for row in read_jsonl(args.predictions)}
        tasks = read_jsonl(args.tasks)
        if args.limit:
            tasks = tasks[:args.limit]
        runner = ExecutorRunner(
            _provider(config), config, Path(args.executor_prompt).read_text(encoding="utf-8")
        )
        results = []
        for task in tasks:
            dialogue_id = task["dialogue_id"]
            dialogue = truth.get(dialogue_id)
            if dialogue is None:
                continue
            prediction = predictions.get(dialogue_id, {})
            output = prediction.get("output") or {}
            induced = output.get("constraints", []) if prediction.get("valid", True) else []
            for probe in dialogue.get("probes", []):
                for background_mode, arm in planned_conditions():
                    for budget in args.budgets:
                        results.append(runner.run_probe(
                            dialogue, probe, induced, background_mode, arm, budget
                        ))
        write_jsonl(args.output, results)
        write_call_logs(args.logs, results)
        result = {
            "probe_n": len(results),
            "scorable_n": sum(row["scorable"] for row in results),
            "status": "engineering_run_not_a_benchmark_result",
        }
        if args.scores:
            write_json(args.scores, score_counterfactual(results))
            result["scores"] = args.scores
    elif args.command == "run-manifest":
        result = build_run_manifest(
            runtime_path=args.runtime, tasks_path=args.tasks, prompt_path=args.prompt,
            predictions_path=args.predictions, logs_path=args.logs, out_dir=args.out_dir,
            contributor=args.contributor, notes=args.notes,
            dimensions_reported=args.dimensions, scores_declared=args.scores_declared,
            repetition_index=args.repetition,
        )
    elif args.command == "validate-submission":
        result = validate_submission(args.dir, benchmark_config_path=args.benchmark_config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["valid"]:
            sys.exit(1)
        return
    elif args.command == "dimensions":
        result = dimension_registry()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
