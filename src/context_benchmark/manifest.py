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

import shutil
from pathlib import Path
from typing import Any

from . import __version__ as PACKAGE_VERSION
from .dimensions import DIMENSION_SET_VERSION, PRIMARY_DIMENSION, validate_dimension_set
from .io import canonical_json, read_json, read_jsonl, sha256_file, sha256_text, write_json

DEFAULT_BENCHMARK_CONFIG = "configs/benchmark.json"


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
    if scores_declared:
        limitations.append("Scores declared by the contributor are not independently reproduced here.")
    if notes:
        limitations.append(notes)

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


def validate_submission(directory: str | Path, *, benchmark_config_path: str | Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    """Validate either a run submission or a dataset extension directory.

    Returns a report; never raises on a bad package (CI needs a readable result).
    """
    root = Path(directory)
    # Validation rules always come from the in-repo canonical config. A caller
    # cannot point this at a hand-edited config to relax version/dimension rules.
    config = benchmark_config(REPO_ROOT / DEFAULT_BENCHMARK_CONFIG)
    issues: list[dict[str, str]] = []
    if not root.is_dir():
        _issue(issues, "not_a_directory", f"{root} does not exist")
        return {"valid": False, "kind": None, "issues": issues, "checked_files": 0}

    if (root / SUBMISSION_FILE).is_file():
        kind = "run_result"
        document = read_json(root / SUBMISSION_FILE)
    elif (root / DATASET_FILE).is_file():
        kind = "dataset_extension"
        document = read_json(root / DATASET_FILE)
    else:
        _issue(issues, "no_manifest", f"neither {SUBMISSION_FILE} nor {DATASET_FILE} found")
        return {"valid": False, "kind": None, "issues": issues, "checked_files": 0}

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

    dataset = document.get("dataset", document)
    if isinstance(dataset, dict):
        if dataset.get("evidence_grade") == "human_gold" and dataset.get("human_annotation_complete") is not True:
            _issue(issues, "fake_human_gold", "human_gold declared without human_annotation_complete=true")
        if dataset.get("evidence_grade") == "human_gold":
            # A boolean alone is self-declared. Require a gold version and its
            # hash to be stamped, so a generator-truth set cannot be relabelled
            # as human gold by flipping one flag.
            if not dataset.get("gold_version"):
                _issue(issues, "human_gold_without_version", "human_gold requires an explicit gold_version")
            evidence = document.get("evidence", {}) if isinstance(document.get("evidence", {}), dict) else {}
            dataset_hash = dataset.get("dataset_manifest_sha256")
            if not dataset_hash and not evidence.get("gold_v1.jsonl"):
                _issue(issues, "human_gold_without_hash", "human_gold requires a dataset_manifest_sha256 or gold_v1.jsonl evidence hash")
        if dataset.get("source_type") == "real_redacted":
            provenance = document.get("provenance", {})
            note = provenance.get("compliance_note") if isinstance(provenance, dict) else None
            if not note or (isinstance(note, str) and len(note.strip()) < 15):
                _issue(issues, "missing_compliance_note", "real_redacted data requires a substantive compliance_note (>=15 chars)")

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
    return {
        "valid": not issues,
        "kind": kind,
        "benchmark_version": document.get("benchmark_version"),
        "submission_id": document.get("submission_id") or document.get("dataset_id"),
        "issues": issues,
        "checked_files": checked,
    }
