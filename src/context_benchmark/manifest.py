"""Submission contract: version stamping, artifact hashing, offline validation.

Two kinds of contributor packages are supported:

* a **run submission directory** containing ``submission.json`` plus the
  prediction/call artifacts it hashes; produced automatically by
  ``run-induction --manifest <dir>`` or by ``run-manifest``;
* a **dataset extension directory** containing ``dataset.json``; produced by
  ``generate --output-dir ...``.

Everything is standard library only. Version stamps are read from
``configs/benchmark.json`` so there is a single source of truth, and every
hash listed in a package is re-checked on validation.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from . import __version__ as PACKAGE_VERSION
from .dimensions import DIMENSION_SET_VERSION, PRIMARY_DIMENSION, validate_dimension_set
from .io import canonical_json, read_json, read_jsonl, sha256_file, sha256_text, write_json
from .validation import validate_output

DEFAULT_BENCHMARK_CONFIG = "configs/benchmark.json"

# Bumped whenever the *rules* of validate_submission change, so a report can be
# traced to the exact contract revision that produced it.
VALIDATOR_VERSION = "submission-validator-v1"
# Recomputed scores are compared with this absolute tolerance.
SCORE_TOLERANCE = 1e-6
# Aggregate metrics the validator is able to recompute from a resolvable
# dataset. A declared score outside this set cannot be verified at all.
RECOMPUTABLE_METRICS = (
    "must_include",
    "must_exclude_absence",
    "fama_macro",
    "empty_output_accuracy",
    "fmr",
    "overgeneralization",
    "unmatched_prediction_rate",
    "invalid_output_rate",
)
# The values a package may put in its scores map.
SCORES_VERIFICATION_VALUES = ("reproduced_locally", "not_reproducible_locally")
# A human_gold claim must ship the gold labels themselves.
GOLD_ARTIFACT_PATTERN = re.compile(r"(^|/)gold[^/]*\.jsonl$", re.IGNORECASE)
# ... and a machine-readable annotation report proving how they were produced.
# Any *report*.json counts, so a package can ship both a human report and an
# AI precheck report; the human one wins and the AI one is never counted.
ANNOTATION_REPORT_PATTERN = re.compile(r"(^|/)[^/]*report[^/]*\.json$", re.IGNORECASE)
# Annotation sources that may never back a human_gold claim.
AI_ANNOTATOR_TYPES = frozenset({
    "ai", "ai_precheck", "ai-precheck", "ai_precheck_only", "ai_generated",
    "ai-generated", "model", "model_generated", "model-generated", "llm",
    "machine", "auto", "automatic", "自动", "模型", "模型生成", "机器",
})


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path; bare relative paths mean the repo's canonical file.

    This makes the CLI work from any cwd (console-script installs) while a
    contributor cannot point the validator at a hand-edited config to relax the
    version rules: validation always uses the canonical in-repo copy.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    in_cwd = Path.cwd() / candidate
    if in_cwd.is_file():
        return in_cwd
    return REPO_ROOT / candidate
SUBMISSION_FILE = "submission.json"
DATASET_FILE = "dataset.json"
FROZEN_DATASET_VERSION = "pilot-v0-candidate"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Evidence entries that are canonical repository files: these MUST hash-match
# the installed tree, otherwise someone can change the scorer/config and ship a
# submission that still claims reproducibility. Other entries (runtime config,
# prompt, tasks.jsonl) are run-specific artefacts copied into the package and
# are verified against the package directory instead.
REPO_EVIDENCE_FILES = frozenset({
    "configs/benchmark.json",
    "configs/generator.json",
    "src/context_benchmark/scoring.py",
})

_BASE_LIMITATIONS = [
    "No human gold and no inter-annotator agreement have been measured for this project yet.",
    "Self-check scores are scorer sanity checks, not model results.",
    "Synthetic template dialogues have low lexical diversity; external validity is unverified.",
]


def benchmark_config(path: str | Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    config = read_json(_resolve_config_path(path))
    if config.get("benchmark_version") is None:
        raise ValueError(f"{path}: benchmark_version missing")
    if config["benchmark_version"] != PACKAGE_VERSION:
        raise ValueError(
            f"version drift: package __version__={PACKAGE_VERSION} but "
            f"{path}:benchmark_version={config['benchmark_version']}"
        )
    return config


def _repo_file(config_path: str | Path, relative: str) -> Path:
    del config_path  # hashes are read from the installed package tree, not the caller's cwd
    return REPO_ROOT / relative


def _sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _frozen_tasks_sha(config: dict[str, Any] | None = None) -> str | None:
    """Hash of the frozen public task file, used to reject copy-paste extensions.

    The hash is stamped into configs/benchmark.json so this check keeps working
    from a pip-installed package in any cwd. The on-disk manifest is only a
    fallback for the in-repo development path.
    """
    if config:
        stamped = (config.get("dataset_policy") or {}).get("frozen_tasks_sha256")
        if stamped:
            return stamped
    path = REPO_ROOT / "data/pilot_v0/manifest.json"
    if not path.is_file():
        return None
    return read_json(path).get("files", {}).get("tasks.jsonl")


def _sum_usage(logs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("prompt_tokens", "completion_tokens", "latency_ms", "cost")
    totals: dict[str, Any] = {field: None for field in fields}
    missing = 0
    for log in logs:
        for field in fields:
            value = log.get(field)
            if value is None:
                missing += 1
                continue
            totals[field] = (totals[field] or 0) + value
    return {"call_n": len(logs), **totals, "missing_usage_fields_n": missing}


def _sum_failures(logs: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "provider_error_n": sum(1 for log in logs if log.get("error")),
        "invalid_output_n": sum(
            1 for log in logs
            if isinstance(log.get("validation"), dict) and not log["validation"].get("valid", False)
        ),
        "failure_counted_n": sum(1 for row in predictions if row.get("failure_counted")),
    }


def build_run_manifest(
    *,
    runtime_path: str | Path,
    tasks_path: str | Path,
    prompt_path: str | Path,
    predictions_path: str | Path,
    logs_path: str | Path,
    out_dir: str | Path,
    benchmark_config_path: str | Path = DEFAULT_BENCHMARK_CONFIG,
    contributor: str = "",
    notes: str = "",
    dimensions_reported: list[str] | None = None,
    scores_declared: bool = False,
    repetition_index: int = 1,
) -> dict[str, Any]:
    """Create a self-contained, version-stamped run submission directory."""
    config = benchmark_config(benchmark_config_path)
    runtime = read_json(runtime_path)
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")
    predictions = read_jsonl(predictions_path)
    logs = read_jsonl(logs_path)

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {
        "predictions.jsonl": Path(predictions_path),
        "calls.jsonl": Path(logs_path),
    }
    for name, source in artifacts.items():
        destination = target / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)

    tasks = Path(tasks_path)
    dataset_root = tasks.parent
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.is_file() else {}

    prompt_sha = sha256_text(prompt_text)
    runtime_sha = sha256_file(runtime_path)
    tasks_sha = sha256_file(tasks)
    scoring_sha = _sha_or_none(_repo_file(benchmark_config_path, "src/context_benchmark/scoring.py"))
    scorer_version = f"scoring.py@{scoring_sha[:12]}" if scoring_sha else "scoring.py@unknown"

    dataset_version = dataset_manifest.get("dataset_version") or dataset_root.name
    run_key = canonical_json({
        "benchmark_version": config["benchmark_version"],
        "dataset_version": dataset_version,
        "model_id": runtime.get("model_id"),
        "tasks_sha256": tasks_sha,
        "runtime_sha256": runtime_sha,
        "repetition_index": repetition_index,
    })
    run_id = f"run-{sha256_text(run_key)[:16]}"

    reported = list(dimensions_reported or [])
    limitations = list(_BASE_LIMITATIONS)
    if dataset_manifest.get("human_annotation_complete") is not True:
        limitations.append("Labels are generator truth checked by a rule audit, not human gold.")
    if notes:
        limitations.append(notes)

    # Declared scores are never taken on trust. When the dataset is reachable
    # here, the package ships the numbers this run's own predictions produce
    # and the validator recomputes them independently from the packaged bytes;
    # when it is not reachable, the package must say so out loud instead of
    # implying a verification that never happened.
    scores_map: dict[str, Any] = {}
    scores_verification: str | None = None
    if scores_declared:
        truth_path = _truth_file(dataset_root)
        dialogues = _load_dialogues(truth_path)
        if dialogues:
            try:
                from .scoring import score_dataset
                aggregate = score_dataset(dialogues, predictions).get("aggregate", {})
            except (KeyError, TypeError, ValueError):
                aggregate = {}
            scores_map = {
                name: (aggregate.get(name) or {}).get("value")
                for name in RECOMPUTABLE_METRICS
                if isinstance(aggregate.get(name), dict)
            }
        if scores_map:
            scores_verification = "reproduced_locally"
        else:
            scores_verification = "not_reproducible_locally"
            limitations.append(
                "The declared scores are not reproducible locally: the dataset behind this run "
                "is not reachable from this machine, so the numbers could not be recomputed "
                "and must not be presented as verified."
            )

    evidence = {
        "configs/benchmark.json": _sha_or_none(_repo_file(benchmark_config_path, "configs/benchmark.json")),
        "configs/generator.json": _sha_or_none(_repo_file(benchmark_config_path, "configs/generator.json")),
        "configs/runtime.json": runtime_sha,
        "prompts/induction_v1.txt": prompt_sha,
        "tasks.jsonl": tasks_sha,
        "dataset/manifest.json": sha256_file(dataset_manifest_path) if dataset_manifest_path.is_file() else None,
        "src/context_benchmark/scoring.py": scoring_sha,
    }

    valid_n = sum(1 for row in predictions if row.get("valid"))
    submission = {
        "schema_version": 1,
        "submission_id": run_id,
        "submission_kind": "run_result",
        "benchmark_version": config["benchmark_version"],
        "dimension_set_version": config["dimension_set_version"],
        "primary_dimension": PRIMARY_DIMENSION,
        "scores_declared": bool(scores_declared),
        "scores_verification": scores_verification,
        "scores": scores_map,
        "dimensions_reported": reported,
        "status": "engineering_run_not_a_benchmark_result",
        "dataset": {
            "dataset_id": dataset_version,
            "dataset_version": dataset_version,
            "dataset_root": str(dataset_root).replace("\\", "/"),
            "split": dataset_manifest.get("split", "public"),
            "evidence_grade": "generator_truth_rule_audited",
            "human_annotation_complete": bool(dataset_manifest.get("human_annotation_complete", False)),
            "gold_version": dataset_manifest.get("gold_version"),
            "source_type": "synthetic_template",
            "generation_method": dataset_manifest.get("generation_method"),
            "seed": dataset_manifest.get("seed"),
            "generator_config_sha256": dataset_manifest.get("generator_config_sha256"),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path) if dataset_manifest_path.is_file() else None,
            "dialogue_n": len(predictions),
        },
        "run": {
            "run_id": run_id,
            "repetition_index": repetition_index,
            "provider": runtime.get("provider"),
            "model": {
                "model_id": runtime.get("model_id"),
                "pinned_version": runtime.get("pinned_version"),
                "weight_sha256": runtime.get("weight_sha256"),
                "quant": runtime.get("quant"),
                "sku": runtime.get("sku"),
            },
            "prompt": {
                "induction_version": runtime.get("prompt_version"),
                "induction_sha256": prompt_sha,
                "runtime_config_sha256": runtime_sha,
            },
            "scorer_version": scorer_version,
            "temperature": runtime.get("temperature"),
            "seed": runtime.get("seed"),
            "usage": _sum_usage(logs),
            "failures": _sum_failures(logs, predictions),
            "denominator_disclosure": {
                "dialogue_n": len(predictions),
                "scored_n": valid_n,
                "failed_n": len(predictions) - valid_n,
            },
        },
        "provenance": {
            "contributor": contributor,
            "license": "MIT",
            "compliance_note": None,
            "submitted_at": None,
        },
        "files": {name: sha256_file(target / name) for name in artifacts},
        "evidence": {key: value for key, value in evidence.items() if value},
        "declared_limitations": limitations,
    }
    write_json(target / SUBMISSION_FILE, submission)
    return {"submission_id": run_id, "path": str(target / SUBMISSION_FILE), "artifact_n": len(artifacts)}


def build_dataset_manifest(
    dataset_dir: str | Path,
    *,
    benchmark_config_path: str | Path = DEFAULT_BENCHMARK_CONFIG,
    contributor: str = "",
    split: str = "public",
    notes: str = "",
) -> dict[str, Any]:
    """Stamp a contributed sample set with its benchmark version and file hashes."""
    config = benchmark_config(benchmark_config_path)
    root = Path(dataset_dir)
    generated = read_json(root / "manifest.json")
    if generated.get("dataset_version") == FROZEN_DATASET_VERSION and root.name != "pilot_v0":
        raise ValueError("An extension set must declare its own dataset_version; pilot-v0-candidate is frozen")

    names = ["tasks.jsonl", "candidate_truth.jsonl", "annotation_worksheet.jsonl",
             "calibration_ids.json", "coverage.json", "manifest.json"]
    present = [name for name in names if (root / name).is_file()]
    files = {name: sha256_file(root / name) for name in present}
    frozen_sha = _frozen_tasks_sha(config)
    if frozen_sha and files.get("tasks.jsonl") == frozen_sha:
        raise ValueError(
            "Extension set reproduces the frozen pilot tasks.jsonl byte-for-byte. "
            "Use a different seed and/or generator parameters; renaming the directory is not a new sample."
        )
    limitations = list(_BASE_LIMITATIONS)
    if notes:
        limitations.append(notes)
    dataset = {
        "schema_version": 1,
        "kind": "dataset_extension",
        "benchmark_version": config["benchmark_version"],
        "dimension_set_version": config["dimension_set_version"],
        "dataset_id": root.name,
        "dataset_version": generated["dataset_version"],
        "split": split,
        "source_type": "synthetic_template",
        "evidence_grade": "generator_truth_rule_audited",
        "human_annotation_complete": bool(generated.get("human_annotation_complete", False)),
        "gold_version": generated.get("gold_version"),
        "seed": generated.get("seed"),
        "generation_method": generated.get("generation_method"),
        "generator_config_sha256": generated.get("generator_config_sha256"),
        "provenance": {"contributor": contributor, "license": "MIT", "compliance_note": None},
        "files": files,
        "declared_limitations": limitations,
    }
    write_json(root / DATASET_FILE, dataset)
    return {"dataset_id": dataset["dataset_id"], "path": str(root / DATASET_FILE), "file_n": len(present)}


def _check_schema_shape(document: dict[str, Any], kind: str, issues: list[dict[str, str]]) -> None:
    """Enforce the JSON Schema contract's required keys and value types.

    The project keeps zero third-party dependencies, so jsonschema is not
    imported. This mirror covers the contract's required fields and the types
    that previous hand-written checks missed (status, provenance, evidence,
    usage/failures structure). It must stay in sync with submission.schema.json.
    """
    required = [
        "schema_version", "submission_id", "submission_kind", "benchmark_version",
        "dimension_set_version", "primary_dimension", "scores_declared",
        "dimensions_reported", "status", "dataset", "run", "provenance",
        "files", "evidence", "declared_limitations",
    ]
    for key in required:
        if key not in document:
            _issue(issues, "missing_required_field", f"submission.json is missing required field {key!r}")
    if document.get("schema_version") != 1:
        _issue(issues, "schema_version_mismatch", "schema_version must be 1")
    if document.get("submission_kind") != kind:
        _issue(issues, "submission_kind_mismatch", f"submission_kind must be {kind!r}")
    if not isinstance(document.get("status"), str) or not document["status"]:
        _issue(issues, "invalid_status", "status must be a non-empty string")
    if not isinstance(document.get("declared_limitations"), list):
        _issue(issues, "invalid_declared_limitations", "declared_limitations must be a list")
    if not isinstance(document.get("scores_declared"), bool):
        _issue(issues, "invalid_scores_declared", "scores_declared must be a boolean")
    verification = document.get("scores_verification")
    if verification is not None and verification not in SCORES_VERIFICATION_VALUES:
        _issue(issues, "invalid_scores_verification",
               f"scores_verification must be one of {list(SCORES_VERIFICATION_VALUES)}, got {verification!r}")
    if document.get("scores_declared"):
        if verification is None:
            _issue(issues, "missing_scores_verification",
                   "a package that declares scores must state scores_verification")
        if verification == "reproduced_locally" and not document.get("scores"):
            _issue(issues, "scores_without_values",
                   "scores_verification='reproduced_locally' requires a non-empty scores map")
    scores = document.get("scores")
    if scores is not None:
        if not isinstance(scores, dict):
            _issue(issues, "invalid_scores", "scores must be an object")
        else:
            for name, value in scores.items():
                if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                    _issue(issues, "invalid_score_value", f"scores.{name} must be a number or null")
    prov = document.get("provenance")
    if not isinstance(prov, dict):
        _issue(issues, "invalid_provenance", "provenance must be an object")
    else:
        for key in ("contributor", "license", "compliance_note"):
            if key not in prov:
                _issue(issues, "missing_provenance_field", f"provenance is missing {key!r}")
    run = document.get("run")
    if isinstance(run, dict):
        for key in ("usage", "failures", "denominator_disclosure"):
            if not isinstance(run.get(key), dict):
                _issue(issues, "invalid_run_detail_type", f"run.{key} must be an object")
        usage = run.get("usage")
        if isinstance(usage, dict):
            for key in ("call_n", "prompt_tokens", "completion_tokens", "latency_ms", "cost"):
                if key not in usage:
                    _issue(issues, "missing_usage_field", f"run.usage is missing {key!r}")
        failures = run.get("failures")
        if isinstance(failures, dict):
            for key in ("provider_error_n", "invalid_output_n", "failure_counted_n"):
                if not isinstance(failures.get(key), int):
                    _issue(issues, "invalid_failure_count", f"run.failures.{key} must be an integer")


def _issue(issues: list[dict[str, str]], code: str, message: str) -> None:
    issues.append({"code": code, "message": message})


def _check_version_fields(document: dict[str, Any], config: dict[str, Any], issues: list[dict[str, str]]) -> None:
    if document.get("benchmark_version") not in config.get("supported_benchmark_versions", []):
        _issue(issues, "unsupported_benchmark_version",
               f"benchmark_version={document.get('benchmark_version')!r} is not in "
               f"{config.get('supported_benchmark_versions')}")
    if document.get("dimension_set_version") != config["dimension_set_version"]:
        _issue(issues, "dimension_set_mismatch",
               f"expected {config['dimension_set_version']!r}, got {document.get('dimension_set_version')!r}")


def _check_hashes(root: Path, files: Any, issues: list[dict[str, str]]) -> int:
    if not isinstance(files, dict) or not files:
        _issue(issues, "no_hashed_artifacts", "files must be a non-empty {path: sha256} map")
        return 0
    checked = 0
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            _issue(issues, "missing_artifact", f"{name} is declared but absent")
            continue
        actual = sha256_file(path)
        if actual != expected:
            _issue(issues, "hash_mismatch", f"{name}: declared {expected[:12]}…, actual {actual[:12]}…")
        checked += 1
    return checked


# ---------------------------------------------------------------------------
# Recomputation helpers: the validator must not take a package's word for
# anything it can recompute from the bytes it ships.
# ---------------------------------------------------------------------------

def _frozen_versions(config: dict[str, Any] | None = None) -> set[str]:
    """Version numbers that may never be re-used for a new gold set."""
    versions = {FROZEN_DATASET_VERSION}
    policy = (config or {}).get("dataset_policy") or {}
    for key in ("frozen_dataset_version", "frozen_versions", "frozen_gold_versions"):
        value = policy.get(key)
        if isinstance(value, str):
            versions.add(value)
        elif isinstance(value, list):
            versions.update(item for item in value if isinstance(item, str))
    return versions


def _tasks_file(directory: Path) -> Path | None:
    path = directory / "tasks.jsonl"
    return path if path.is_file() else None


def _truth_file(directory: Path) -> Path | None:
    for name in ("candidate_truth.jsonl", "truth.jsonl", "gold.jsonl", "gold_v1.jsonl"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _resolve_dataset_dir(root: Path, dataset: dict[str, Any]) -> Path | None:
    """Find the dataset a package was scored on, if it is reachable here.

    A package may point at an absolute path on the machine that produced it,
    a path relative to the package, or carry the dataset itself. When none of
    those resolve, the scores in the package simply cannot be recomputed —
    which the package then has to declare.
    """
    candidates: list[Path] = []
    raw = dataset.get("dataset_root")
    if isinstance(raw, str) and raw.strip():
        declared = Path(raw)
        if declared.is_absolute():
            candidates.append(declared)
        else:
            candidates.extend([root / declared, Path.cwd() / declared, REPO_ROOT / declared])
    candidates.extend([root / "dataset", root])
    for candidate in candidates:
        try:
            if candidate.is_dir() and _tasks_file(candidate) and _truth_file(candidate):
                return candidate
        except OSError:
            continue
    return None


def _load_dialogues(truth_path: Path | None) -> list[dict[str, Any]]:
    """Read a truth/gold file into scorer-shaped dialogues."""
    if truth_path is None:
        return []
    try:
        rows = read_jsonl(truth_path)
    except (ValueError, OSError):
        return []
    dialogues: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gold = row.get("candidate_truth")
        if not isinstance(gold, list) and isinstance(row.get("constraints"), list):
            gold = row["constraints"]
        if not isinstance(gold, list):
            continue
        dialogues.append({
            "dialogue_id": row.get("dialogue_id"),
            "turns": row.get("turns") or [],
            "candidate_truth": gold,
            "distractors": row.get("distractors") or [],
        })
    return dialogues


def _packaged_gold(files: dict[str, Any]) -> str | None:
    for name in sorted(files):
        if GOLD_ARTIFACT_PATTERN.search(str(name).replace("\\", "/")):
            return str(name)
    return None


def _row_output(row: Any) -> Any:
    """The output payload of a prediction row, mirroring the scorer's fallback."""
    if not isinstance(row, dict):
        return None
    if "output" in row:
        return row["output"]
    return {"constraints": row.get("constraints", [])}


def _recompute_denominators(rows: list[Any], turns_by_id: dict[str, list[Any]]) -> dict[str, Any]:
    """Recount scored/failed rows from the packaged predictions themselves.

    ``invalid`` is the union of two independent signals, counted per row:

    * the row's output does not satisfy the output contract (shape check, and
      the full semantic check when the dialogue turns are available), and
    * the row declares itself as a failure (``valid`` is false, or
      ``failure_counted`` is true).

    A package can therefore neither inflate ``scored_n`` by shipping broken
    outputs, nor hide a failure behind a hand-written ``valid: true``.

    When the benchmark dataset is reachable, the *semantic* count is the
    authoritative one (it is what the runner itself scored with); the shape
    count is reported next to it so a disagreement stays visible.
    """
    total = len(rows)
    shape_invalid = 0
    full_invalid = 0
    declared_invalid = 0
    base_invalid = 0
    authoritative_invalid = 0
    for row in rows:
        if not isinstance(row, dict):
            shape_invalid += 1
            full_invalid += 1
            declared_invalid += 1
            base_invalid += 1
            authoritative_invalid += 1
            continue
        output = _row_output(row)
        shape_ok = isinstance(output, dict) and validate_output(output, None).valid
        if not shape_ok:
            shape_invalid += 1
        turns = turns_by_id.get(row.get("dialogue_id"))
        full_ok = shape_ok and validate_output(output, turns).valid if turns is not None else shape_ok
        if not full_ok:
            full_invalid += 1
        # ``valid`` mirrors the runner's own bookkeeping: a missing key means
        # the row was never confirmed, so it cannot be counted as scored.
        row_declared_invalid = not bool(row.get("valid")) or row.get("failure_counted") is True
        if row_declared_invalid:
            declared_invalid += 1
        if not shape_ok or row_declared_invalid:
            base_invalid += 1
        if not full_ok or row_declared_invalid:
            authoritative_invalid += 1
    semantic_available = bool(turns_by_id)
    invalid_n = authoritative_invalid if semantic_available else base_invalid
    return {
        "row_n": total,
        "invalid_shape_n": shape_invalid,
        "invalid_full_n": full_invalid,
        "invalid_declared_n": declared_invalid,
        "invalid_base_n": base_invalid,
        "invalid_n": invalid_n,
        "scored_n": total - invalid_n,
        "failed_n": invalid_n,
        "semantic_check_available": semantic_available,
    }


def _normalize_kappa_map(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        number: Any = None
        if isinstance(item, dict):
            for candidate in ("kappa", "value", "ac1", "gwet_ac1"):
                if isinstance(item.get(candidate), (int, float)) and not isinstance(item.get(candidate), bool):
                    number = item[candidate]
                    break
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            number = item
        if number is not None:
            result[str(key)] = float(number)
    return result


def _annotation_facts(report: dict[str, Any]) -> dict[str, Any]:
    """Extract the facts a human_gold annotation report has to state.

    Both the flat spelling used by this contract and the nested spelling of
    ``agreement.agreement_report`` are accepted, so an existing agreement
    report can be shipped as-is.
    """
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    facts: dict[str, Any] = {}

    kappa = report.get("overall_kappa", report.get("kappa", overall.get("kappa")))
    facts["overall_kappa"] = float(kappa) if isinstance(kappa, (int, float)) and not isinstance(kappa, bool) else None

    facts["per_class_kappa"] = _normalize_kappa_map(
        report.get("per_class_kappa", report.get("per_type_kappa", report.get("per_type", report.get("per_class"))))
    )

    ci = report.get("kappa_ci_lower", report.get("ci_95_lower", overall.get("kappa_ci_lower")))
    if ci is None:
        bootstrap = overall.get("ci_95_percentile_cluster_bootstrap")
        if isinstance(bootstrap, list) and bootstrap and isinstance(bootstrap[0], (int, float)):
            ci = bootstrap[0]
    if isinstance(ci, dict):
        ci = ci.get("overall")
    facts["kappa_ci_lower"] = float(ci) if isinstance(ci, (int, float)) and not isinstance(ci, bool) else None

    ac1 = report.get("ac1", report.get("gwet_ac1", report.get("overall_ac1", overall.get("ac1"))))
    if isinstance(ac1, dict):
        ac1 = ac1.get("overall")
    facts["ac1"] = float(ac1) if isinstance(ac1, (int, float)) and not isinstance(ac1, bool) else None

    unit_n = report.get(
        "unit_n", report.get("n_units", report.get("annotated_unit_n", report.get("aligned_unit_n", overall.get("n"))))
    )
    facts["unit_n"] = unit_n if isinstance(unit_n, int) and not isinstance(unit_n, bool) else None

    annotators = report.get("annotator_n", report.get("n_annotators"))
    if annotators is None:
        raw_annotators = report.get("annotators")
        if isinstance(raw_annotators, list):
            annotators = len(raw_annotators)
        elif isinstance(raw_annotators, int) and not isinstance(raw_annotators, bool):
            annotators = raw_annotators
    facts["annotator_n"] = annotators if isinstance(annotators, int) and not isinstance(annotators, bool) else None

    adjudication = report.get("adjudication_method", report.get("adjudication"))
    if isinstance(adjudication, dict):
        adjudication = adjudication.get("method")
    facts["adjudication_method"] = adjudication.strip() if isinstance(adjudication, str) and adjudication.strip() else None

    annotator_type = report.get("annotator_type")
    facts["annotator_type"] = str(annotator_type).strip().lower() if isinstance(annotator_type, str) and annotator_type.strip() else None
    facts["sources"] = report.get("sources") if isinstance(report.get("sources"), list) else []
    return facts


def _check_human_gold(
    root: Path,
    document: dict[str, Any],
    dataset: dict[str, Any],
    files: dict[str, Any],
    config: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """Hard gate for ``evidence_grade == "human_gold"``.

    A human gold claim is only accepted when the package actually ships the
    gold labels, ships an annotation report that states how they were
    produced, and the gold version is not a recycled frozen version.
    """
    info: dict[str, Any] = {"checked": True, "verified": False, "gold_file": None, "annotation_report": None}
    start = len(issues)

    if dataset.get("human_annotation_complete") is not True:
        _issue(issues, "fake_human_gold", "human_gold declared without human_annotation_complete=true")

    gold_version = dataset.get("gold_version")
    if not gold_version:
        _issue(issues, "human_gold_without_version", "human_gold requires an explicit gold_version")
    elif gold_version in _frozen_versions(config):
        _issue(issues, "human_gold_reuses_frozen_version",
               f"gold_version {gold_version!r} reuses a frozen version and is not a new gold set")

    gold_name = _packaged_gold(files)
    if gold_name is None:
        _issue(issues, "human_gold_missing_artifacts",
               "human_gold requires a gold artifact (e.g. gold.jsonl) declared in files")
        _issue(issues, "human_gold_without_hash",
               "human_gold requires the gold artifact hash to be declared")
    else:
        info["gold_file"] = gold_name
        gold_path = root / gold_name
        if not gold_path.is_file():
            _issue(issues, "human_gold_missing_artifacts", f"gold artifact {gold_name} is absent from the package")
        else:
            actual = sha256_file(gold_path)
            declared = files.get(gold_name)
            if isinstance(declared, str) and actual != declared:
                _issue(issues, "human_gold_hash_mismatch",
                       f"{gold_name}: declared {declared[:12]}…, actual {actual[:12]}…")
            declared_gold_sha = dataset.get("gold_sha256")
            if isinstance(declared_gold_sha, str) and declared_gold_sha != actual:
                _issue(issues, "human_gold_hash_mismatch",
                       f"{gold_name}: dataset.gold_sha256 does not match the packaged gold file")

    report_name = dataset.get("annotation_report_file")
    if isinstance(report_name, str) and report_name:
        candidates = [report_name]
    else:
        candidates = sorted(
            str(name) for name in files
            if ANNOTATION_REPORT_PATTERN.search(str(name).replace("\\", "/"))
        )
    human_report: tuple[str, dict[str, Any]] | None = None
    ai_reports: list[str] = []
    for name in candidates:
        path = root / name
        if not path.is_file():
            _issue(issues, "human_gold_missing_artifacts", f"annotation report {name} is absent from the package")
            continue
        try:
            data = read_json(path)
        except (ValueError, OSError):
            _issue(issues, "human_gold_report_invalid", f"{name} is not valid JSON")
            continue
        if not isinstance(data, dict):
            _issue(issues, "human_gold_report_invalid", f"{name} must be a JSON object")
            continue
        facts = _annotation_facts(data)
        source = facts["annotator_type"]
        if source in AI_ANNOTATOR_TYPES:
            ai_reports.append(name)
            continue
        if human_report is None:
            human_report = (name, facts)

    if human_report is None:
        if ai_reports:
            _issue(issues, "human_gold_annotator_not_human",
                   f"annotation report {ai_reports[0]} is not human-annotated and cannot back a human_gold claim")
        else:
            _issue(issues, "human_gold_missing_artifacts",
                   "human_gold requires an annotation report (e.g. annotation_report.json) declared in files")
    else:
        name, facts = human_report
        info["annotation_report"] = name
        missing: list[str] = []
        if facts["annotator_type"] != "human":
            missing.append("annotator_type=human")
        if facts["annotator_n"] is None:
            missing.append("annotator_n")
        elif facts["annotator_n"] < 2:
            _issue(issues, "human_gold_insufficient_annotators",
                   f"{name}: annotator_n={facts['annotator_n']}; inter-annotator agreement needs at least two annotators")
        if facts["unit_n"] is None or facts["unit_n"] <= 0:
            missing.append("unit_n")
        if facts["overall_kappa"] is None:
            missing.append("overall_kappa")
        if not facts["per_class_kappa"]:
            missing.append("per_class_kappa")
        if facts["kappa_ci_lower"] is None:
            missing.append("kappa_ci_lower")
        if facts["ac1"] is None:
            missing.append("ac1")
        if facts["adjudication_method"] is None:
            missing.append("adjudication_method")
        if missing:
            _issue(issues, "human_gold_report_incomplete", f"{name} is missing: {', '.join(missing)}")

        sources = facts["sources"]
        if sources:
            human_sources = [
                item for item in sources
                if isinstance(item, dict) and str(item.get("type", "")).strip().lower() not in AI_ANNOTATOR_TYPES
            ]
            ai_sources = [
                item for item in sources
                if isinstance(item, dict) and str(item.get("type", "")).strip().lower() in AI_ANNOTATOR_TYPES
            ]
            if not human_sources:
                _issue(issues, "human_gold_annotator_not_human",
                       f"{name}: no human annotation source; AI prechecks cannot back a human_gold claim")
            else:
                human_units = sum(
                    int(item.get("unit_n", 0)) for item in human_sources
                    if isinstance(item.get("unit_n"), int) and not isinstance(item.get("unit_n"), bool)
                )
                ai_units = sum(
                    int(item.get("unit_n", 0)) for item in ai_sources
                    if isinstance(item.get("unit_n"), int) and not isinstance(item.get("unit_n"), bool)
                )
                if ai_units and facts["unit_n"] is not None and facts["unit_n"] > human_units:
                    _issue(issues, "human_gold_ai_source_counted",
                           f"{name}: {ai_units} AI-sourced unit(s) are counted in the human annotation total")
                for item in human_sources:
                    value = item.get("overall_kappa", item.get("kappa"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and facts["overall_kappa"] is not None:
                        if abs(float(value) - facts["overall_kappa"]) > 1e-9:
                            _issue(issues, "human_gold_source_conflict",
                                   f"{name}: the declared kappa disagrees with the human annotator's kappa")

    info["verified"] = len(issues) == start
    return info


def _mentions_reproducibility(limitations: Any) -> bool:
    if not isinstance(limitations, list):
        return False
    for item in limitations:
        if not isinstance(item, str):
            continue
        lowered = item.lower()
        if "reproduc" in lowered or "复现" in item:
            return True
    return False


def validate_submission(directory: str | Path, *, benchmark_config_path: str | Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    """Validate either a run submission or a dataset extension directory.

    Returns a report; never raises on a bad package (CI needs a readable result).
    """
    root = Path(directory)
    # Validation rules always come from the in-repo canonical config. A caller
    # cannot point this at a hand-edited config to relax version/dimension rules.
    config = benchmark_config(REPO_ROOT / DEFAULT_BENCHMARK_CONFIG)
    issues: list[dict[str, str]] = []
    base_report: dict[str, Any] = {
        "validator_version": VALIDATOR_VERSION,
        "benchmark_config_source": "repo_canonical",
        "benchmark_config_path_argument": "ignored_by_design",
        "scores_verified": False,
        "denominators_recomputed": False,
        "human_gold_verified": False,
    }
    if not root.is_dir():
        _issue(issues, "not_a_directory", f"{root} does not exist")
        return {**base_report, "valid": False, "kind": None, "issues": issues, "checked_files": 0}

    if (root / SUBMISSION_FILE).is_file():
        kind = "run_result"
        document = read_json(root / SUBMISSION_FILE)
    elif (root / DATASET_FILE).is_file():
        kind = "dataset_extension"
        document = read_json(root / DATASET_FILE)
    else:
        _issue(issues, "no_manifest", f"neither {SUBMISSION_FILE} nor {DATASET_FILE} found")
        return {**base_report, "valid": False, "kind": None, "issues": issues, "checked_files": 0}

    _check_version_fields(document, config, issues)
    if kind == "run_result":
        _check_schema_shape(document, kind, issues)
    if kind == "dataset_extension":
        if document.get("dataset_version") == FROZEN_DATASET_VERSION and root.name != "pilot_v0":
            _issue(issues, "frozen_dataset_version_reused",
                   "an extension set cannot reuse the frozen pilot-v0-candidate version")
        frozen_sha = _frozen_tasks_sha(config)
        if frozen_sha and document.get("files", {}).get("tasks.jsonl") == frozen_sha:
            _issue(issues, "duplicate_of_frozen_pilot",
                   "tasks.jsonl is byte-identical to the frozen pilot set; this is not a new sample")

    files = document.get("files") if isinstance(document.get("files"), dict) else {}
    dataset = document.get("dataset", document)
    if not isinstance(dataset, dict):
        dataset = {}
    if dataset.get("source_type") == "real_redacted":
        provenance = document.get("provenance", {})
        note = provenance.get("compliance_note") if isinstance(provenance, dict) else None
        if not note or (isinstance(note, str) and len(note.strip()) < 15):
            _issue(issues, "missing_compliance_note", "real_redacted data requires a substantive compliance_note (>=15 chars)")

    human_gold_info: dict[str, Any] = {"checked": False, "verified": False}
    if dataset.get("evidence_grade") == "human_gold":
        human_gold_info = _check_human_gold(root, document, dataset, files, config, issues)

    # The dataset a package was scored on. Without it, denominators can only be
    # recomputed from the packaged predictions, and scores cannot be checked.
    dataset_dir = _resolve_dataset_dir(root, dataset)
    truth_path = _truth_file(dataset_dir) if dataset_dir else None
    packaged_gold = _packaged_gold(files)
    if dataset.get("evidence_grade") == "human_gold" and packaged_gold:
        truth_path = root / packaged_gold
    dialogues = _load_dialogues(truth_path)
    turns_by_id = {
        dialogue["dialogue_id"]: dialogue["turns"]
        for dialogue in dialogues if dialogue.get("dialogue_id")
    }

    denominator_report: dict[str, Any] = {"recomputed": False}
    score_report: dict[str, Any] = {"verified": False}
    declared_scores = document.get("scores")
    declared_scores = declared_scores if isinstance(declared_scores, dict) else {}
    prediction_rows: list[Any] = []
    predictions_path: Path | None = None
    if kind == "run_result":
        for name in sorted(files):
            if str(name).replace("\\", "/").rsplit("/", 1)[-1] == "predictions.jsonl":
                predictions_path = root / str(name)
                break
        if predictions_path is None or not predictions_path.is_file():
            _issue(issues, "predictions_missing",
                   "a run submission must ship predictions.jsonl so the denominators can be recomputed")
        else:
            try:
                prediction_rows = read_jsonl(predictions_path)
            except (ValueError, OSError) as exc:
                prediction_rows = []
                _issue(issues, "predictions_unreadable", f"predictions.jsonl could not be parsed: {exc}")
            recomputed = _recompute_denominators(prediction_rows, turns_by_id)
            denominator_report = {
                "recomputed": True,
                "source": str(predictions_path.relative_to(root)) if predictions_path.is_relative_to(root) else str(predictions_path),
                "row_n": recomputed["row_n"],
                "invalid_n": recomputed["invalid_n"],
                "invalid_shape_n": recomputed["invalid_shape_n"],
                "invalid_full_n": recomputed["invalid_full_n"],
                "invalid_declared_n": recomputed["invalid_declared_n"],
                "invalid_base_n": recomputed["invalid_base_n"],
                "scored_n": recomputed["scored_n"],
                "failed_n": recomputed["failed_n"],
                "semantic_check_available": recomputed["semantic_check_available"],
            }
            declared_denominator = document.get("run", {}).get("denominator_disclosure", {})
            if not isinstance(declared_denominator, dict):
                declared_denominator = {}
            declared_dialogue_n = declared_denominator.get("dialogue_n")
            declared_scored_n = declared_denominator.get("scored_n")
            declared_failed_n = declared_denominator.get("failed_n")
            if declared_dialogue_n != recomputed["row_n"]:
                _issue(issues, "denominator_mismatch",
                       f"dialogue_n={declared_dialogue_n!r} but predictions.jsonl holds {recomputed['row_n']} rows")
            # The declared failure count must cover every row the validator can
            # prove broken. When the dataset is reachable, the semantic
            # re-validation is authoritative and must match exactly; without
            # turns only the shape check is available, so the declared count
            # may be larger (semantic failures are invisible here) but it may
            # never be smaller than what the shape check proves broken.
            if recomputed["semantic_check_available"]:
                if declared_failed_n != recomputed["failed_n"] or declared_scored_n != recomputed["scored_n"]:
                    _issue(issues, "denominator_mismatch",
                           f"declared scored_n/failed_n={declared_scored_n!r}/{declared_failed_n!r}, "
                           f"recomputed {recomputed['scored_n']}/{recomputed['failed_n']} from predictions.jsonl")
            else:
                if not isinstance(declared_failed_n, int) or declared_failed_n < recomputed["invalid_base_n"]:
                    _issue(issues, "denominator_mismatch",
                           f"declared failed_n={declared_failed_n!r} but {recomputed['invalid_base_n']} packaged "
                           f"prediction(s) fail the output contract")
                if not isinstance(declared_scored_n, int) or declared_scored_n > recomputed["row_n"] - recomputed["invalid_base_n"]:
                    _issue(issues, "denominator_mismatch",
                           f"declared scored_n={declared_scored_n!r} exceeds the "
                           f"{recomputed['row_n'] - recomputed['invalid_base_n']} packaged prediction(s) "
                           f"that satisfy the output contract")

    # Scores: recompute them from the dataset when it is reachable, otherwise
    # the package has to say out loud that its scores cannot be reproduced.
    # Nothing here ever takes the package's word for a number it can check.
    recomputed_aggregate: dict[str, Any] | None = None
    if kind == "run_result" and dialogues and prediction_rows:
        from .scoring import score_dataset
        try:
            aggregate = score_dataset(dialogues, prediction_rows)["aggregate"]
        except (KeyError, TypeError, ValueError) as exc:
            aggregate = {}
            _issue(issues, "scores_unrecomputable", f"the dataset could not be re-scored: {exc}")
        recomputed_aggregate = {
            name: (aggregate.get(name) or {}).get("value")
            for name in RECOMPUTABLE_METRICS
            if isinstance(aggregate.get(name), dict)
        }

    scores_claimed = bool(declared_scores) or bool(document.get("scores_declared"))
    score_report = {
        "verified": False,
        "scores_claimed": scores_claimed,
        "declared_metrics": sorted(declared_scores),
    }
    if scores_claimed and recomputed_aggregate is None:
        # The numbers cannot be checked here, so the package must say so.
        if document.get("scores_verification") != "not_reproducible_locally":
            _issue(issues, "scores_not_reproducible_locally",
                   "scores are declared but cannot be reproduced here; declare "
                   "scores_verification='not_reproducible_locally'")
        if not _mentions_reproducibility(document.get("declared_limitations")):
            _issue(issues, "scores_limitation_missing",
                   "declared_limitations must state that the declared scores are not reproducible locally")
        score_report["reason"] = "dataset_not_resolvable"
    elif scores_claimed and recomputed_aggregate is not None:
        if not declared_scores:
            _issue(issues, "scores_missing",
                   "scores_declared is true but the package ships no scores to verify")
        mismatched: list[str] = []
        unverifiable: list[str] = []
        malformed: list[str] = []
        for name, value in sorted(declared_scores.items()):
            if name not in RECOMPUTABLE_METRICS:
                unverifiable.append(name)
                continue
            computed = recomputed_aggregate.get(name)
            if value is None and computed is None:
                continue
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                malformed.append(name)
                continue
            if value is None or computed is None:
                mismatched.append(name)
                continue
            if abs(float(value) - float(computed)) > SCORE_TOLERANCE:
                mismatched.append(name)
        for name in unverifiable:
            _issue(issues, "unverifiable_score_metric",
                   f"scores.{name} cannot be recomputed by the validator")
        for name in malformed:
            _issue(issues, "invalid_score_value", f"scores.{name} must be a number or null")
        for name in mismatched:
            _issue(issues, "score_mismatch",
                   f"scores.{name} does not match the recomputed value")
        verified = not mismatched and not unverifiable and not malformed
        score_report.update({
            "verified": verified,
            "reason": "recomputed" if verified else "mismatch",
            "tolerance": SCORE_TOLERANCE,
            "mismatched": mismatched,
            "unverifiable": unverifiable,
            "malformed": malformed,
            "recomputed": recomputed_aggregate,
        })
    elif document.get("scores_verification") is not None and document.get("scores_verification") != "not_reproducible_locally":
        _issue(issues, "scores_without_scores",
               "scores_verification is declared but no scores are shipped to verify")
    elif recomputed_aggregate is not None:
        score_report["recomputed"] = recomputed_aggregate

    if kind == "run_result":
        run = document.get("run", {})
        model = run.get("model", {})
        if not model.get("model_id") or not model.get("pinned_version"):
            _issue(issues, "unpinned_model", "model_id and pinned_version are mandatory (fail closed)")
        denom = run.get("denominator_disclosure", {})
        if sum(denom.get(key, 0) for key in ("scored_n", "failed_n")) != denom.get("dialogue_n"):
            _issue(issues, "denominator_mismatch", "scored_n + failed_n must equal dialogue_n")
        if not run.get("usage") or not run.get("failures"):
            _issue(issues, "missing_run_detail", "usage and failures are mandatory")
        reported = document.get("dimensions_reported", [])
        for code in validate_dimension_set(reported, scores_declared=bool(document.get("scores_declared"))):
            _issue(issues, code, "dimension set violation")
        if document.get("primary_dimension") != PRIMARY_DIMENSION:
            _issue(issues, "primary_dimension_mismatch", f"primary dimension must be {PRIMARY_DIMENSION}")

    checked = _check_hashes(root, document.get("files"), issues)
    # The evidence map records the hashes of the code/config/prompts that
    # produced the score. Without re-checking it, anyone can edit scoring.py
    # (or the benchmark config) and ship a "reproducible" submission with stale
    # or invented evidence hashes. Verify against the installed repo tree.
    evidence = document.get("evidence")
    if kind == "run_result":
        if not isinstance(evidence, dict) or not evidence:
            _issue(issues, "no_evidence", "a run submission needs a non-empty {path: sha256} evidence map")
        else:
            for name, expected in evidence.items():
                if name not in REPO_EVIDENCE_FILES:
                    # Run-specific provenance (runtime, prompt, tasks, dataset
                    # manifest) is informational here; those bytes are verified
                    # through files{} and the dataset manifest hash elsewhere.
                    continue
                path = REPO_ROOT / name
                if not path.is_file():
                    _issue(issues, "missing_evidence_file", f"evidence file {name} is absent from the benchmark tree")
                    continue
                actual = sha256_file(path)
                if actual != expected:
                    _issue(issues, "evidence_hash_mismatch", f"{name}: declared {str(expected)[:12]}…, actual {actual[:12]}…")
    # The scorer version must match the actual scoring.py content, otherwise a
    # submission can claim a scorer it was never scored by.
    scorer_version = (document.get("run") or {}).get("scorer_version")
    scoring_path = REPO_ROOT / "src/context_benchmark/scoring.py"
    if kind == "run_result" and scoring_path.is_file():
        expected_scorer = f"scoring.py@{sha256_file(scoring_path)[:12]}"
        if scorer_version != expected_scorer:
            _issue(issues, "scorer_version_mismatch", f"declared {scorer_version!r}, expected {expected_scorer!r}")

    declared_validator_version = document.get("validator_version")
    if declared_validator_version is not None and declared_validator_version != VALIDATOR_VERSION:
        _issue(issues, "validator_version_mismatch",
               f"package targets validator {declared_validator_version!r}, this validator is {VALIDATOR_VERSION!r}")

    return {
        **base_report,
        "valid": not issues,
        "kind": kind,
        "benchmark_version": document.get("benchmark_version"),
        "submission_id": document.get("submission_id") or document.get("dataset_id"),
        "issues": issues,
        "checked_files": checked,
        "scores_verified": bool(score_report.get("verified")),
        "denominators_recomputed": bool(denominator_report.get("recomputed")),
        "human_gold_verified": bool(human_gold_info.get("verified")),
        "denominator_recomputation": denominator_report,
        "score_verification": score_report,
        "human_gold_verification": human_gold_info,
    }

