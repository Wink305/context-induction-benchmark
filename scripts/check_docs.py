"""Single-source-of-truth check for the numbers quoted in the documentation.

The audit found that the dataset SHA-256 and the test count were quoted
differently in README.md, P0_标注规范与试点方案.md and data/pilot_v0/manifest.json.
This script recomputes every quoted value and fails when a document drifts.

Run: python scripts/check_docs.py
Exit code 1 on any inconsistency.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_TRUTH = ROOT / "data" / "pilot_v0" / "candidate_truth.jsonl"
TASKS = ROOT / "data" / "pilot_v0" / "tasks.jsonl"
README = ROOT / "README.md"
P0_DOC = ROOT / "P0_标注规范与试点方案.md"
MANIFEST = ROOT / "data" / "pilot_v0" / "manifest.json"

problems: list[str] = []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def count_tests() -> int:
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        total += len(re.findall(r"^\s*def test_", source, re.MULTILINE))
    return total


def check(label: str, condition: bool, detail: str) -> None:
    if not condition:
        problems.append(f"{label}: {detail}")


def main() -> int:
    truth_sha = sha256_file(CANDIDATE_TRUTH)
    tasks_sha = sha256_file(TASKS)
    tests_n = count_tests()

    readme = README.read_text(encoding="utf-8")
    p0 = P0_DOC.read_text(encoding="utf-8")

    # 1. Dataset hashes quoted in the documents must match the files on disk.
    for name, text in (("README.md", readme), ("P0_标注规范与试点方案.md", p0)):
        for quoted in set(re.findall(r"\b[0-9a-f]{8,64}\b", text)):
            if len(quoted) >= 40 and quoted not in {truth_sha, tasks_sha}:
                # Only complain when the surrounding text claims it is the
                # dataset hash.
                for context in re.finditer(rf"[^\n]*{quoted}[^\n]*", text):
                    line = context.group(0)
                    if re.search(r"(sha|hash|哈希|SHA-256)", line, re.IGNORECASE) and quoted not in {truth_sha, tasks_sha}:
                        problems.append(
                            f"{name}: quoted hash {quoted[:12]}… does not match any current artifact "
                            f"(candidate_truth={truth_sha[:12]}…, tasks={tasks_sha[:12]}…) — line: {line.strip()[:90]}"
                        )

    # 2. The frozen tasks hash constant in configs/benchmark.json must be current.
    benchmark_config = (ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8")
    declared = re.search(r'"frozen_tasks_sha256":\s*"([0-9a-f]{64})"', benchmark_config)
    check(
        "configs/benchmark.json",
        bool(declared) and declared.group(1) == tasks_sha,
        f"frozen_tasks_sha256={declared.group(1)[:12] if declared else 'missing'}… but tasks.jsonl is {tasks_sha[:12]}…",
    )

    # 3. Every quoted test count must match the real number of test methods.
    #    Sections stamped with a date are historical snapshots and are exempt.
    for name, text in (("README.md", readme), ("P0_标注规范与试点方案.md", p0)):
        for quoted in set(re.findall(r"(\d{2,4})\s*(?:\+|项|个)?\s*(?:项|个)?\s*(?:tests?|单元测试|tests)", text, re.IGNORECASE)):
            value = int(quoted)
            if 50 <= value <= 1000 and value != tests_n and not (value < tests_n and "+" in text):
                problems.append(
                    f"{name}: quotes {value} tests but the suite currently has {tests_n} test methods"
                )

    # 4. The manifest must record the same dataset version as the documents claim.
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    version = re.search(r'"dataset_version":\s*"([^"]+)"', manifest_text)
    check("data/pilot_v0/manifest.json", bool(version), "dataset_version missing")
    if version:
        check(
            "README.md",
            version.group(1) in readme,
            f"dataset version {version.group(1)} is not mentioned in README.md",
        )

    # 5. Prompt file names must match the versions they declare.
    for prompt in sorted((ROOT / "prompts").glob("*.txt")):
        first = prompt.read_text(encoding="utf-8").splitlines()[0]
        declared_version = re.search(r"PROMPT_VERSION:\s*(\S+)", first)
        if not declared_version:
            problems.append(f"{prompt.name}: no PROMPT_VERSION header")
            continue
        # "induction-v1" and "executor-v2" -> suffix "v1" / "v2"
        suffix = declared_version.group(1).rsplit("-", 1)[-1]
        stem_version = re.search(r"_(v\d+)\.txt$", prompt.name)
        if stem_version and stem_version.group(1) != suffix:
            problems.append(
                f"{prompt.name}: declares PROMPT_VERSION {declared_version.group(1)} but the file name says {stem_version.group(1)}"
            )

    # 6. The machine-readable implementation_status must mirror the README
    #    table, and every "implemented" claim must point at files that exist.
    config = json.loads((ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8"))
    status = config.get("implementation_status") or {}
    entries = status.get("entries") or []
    check(
        "configs/benchmark.json",
        bool(entries),
        "implementation_status.entries is missing — the README claims this table is machine-readable",
    )
    for entry in entries:
        promise = entry.get("promise", "")
        state = entry.get("state", "")
        check(
            "configs/benchmark.json",
            state in {"implemented", "partial", "not_implemented"},
            f"implementation_status[{entry.get('id')}].state={state!r} is not a legal state",
        )
        if promise:
            check(
                "README.md",
                promise in readme,
                f"implementation_status[{entry.get('id')}] promise {promise[:40]!r} is not in the README table",
            )
        for evidence_path in entry.get("evidence", []):
            check(
                f"configs/benchmark.json[{entry.get('id')}]",
                (ROOT / evidence_path).exists(),
                f"evidence path {evidence_path!r} does not exist",
            )
        if state == "implemented" and not entry.get("evidence"):
            problems.append(
                f"implementation_status[{entry.get('id')}]: state 'implemented' but no evidence is listed"
            )

    # 7. The README must not claim a pilot probe set that lacks the
    #    false-trigger control. The frozen pilot has 70 probes, all
    #    related=true; the randomised extension adds 35 related=false ones.
    contrib_dir = ROOT / "data" / "contrib" / "pilot-v1-randomized" / "candidate_truth.jsonl"
    if contrib_dir.is_file():
        related_false = 0
        related_true = 0
        for line in contrib_dir.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for probe in json.loads(line).get("probes", []) or []:
                if probe.get("related") is False:
                    related_false += 1
                else:
                    related_true += 1
        check(
            "data/contrib/pilot-v1-randomized",
            related_false > 0,
            "the randomised extension set has no related=false probes, so dimension B still has no false-trigger estimate",
        )

    # 8. The committed self-check artifact must quote the same scorer version
    #    the code declares, so a stale artifact cannot masquerade as current.
    self_check_path = ROOT / "data" / "pilot_v0" / "self_check.json"
    if self_check_path.is_file():
        artifact = json.loads(self_check_path.read_text(encoding="utf-8"))
        scoring_source = (ROOT / "src" / "context_benchmark" / "scoring.py").read_text(encoding="utf-8")
        matcher = re.search(r'MATCHER_VERSION\s*=\s*"([^"]+)"', scoring_source)
        scorer = re.search(r'SCORER_VERSION\s*=\s*"([^"]+)"', (ROOT / "src" / "context_benchmark" / "cli.py").read_text(encoding="utf-8"))
        if matcher:
            check(
                "data/pilot_v0/self_check.json",
                artifact.get("matcher_version") == matcher.group(1),
                f"matcher_version={artifact.get('matcher_version')!r} but scoring.py declares {matcher.group(1)!r}",
            )
        if scorer:
            check(
                "data/pilot_v0/self_check.json",
                artifact.get("scorer_version") == scorer.group(1),
                f"scorer_version={artifact.get('scorer_version')!r} but cli.py declares {scorer.group(1)!r}",
            )

    # 9. The annotator manual and the rendered sheet must describe the same
    #    answer-block format.  A column added to one and not the other sends
    #    annotators a form the parser cannot read, and the boundary-tag
    #    column is what the agreement gate depends on.
    manual_path = ROOT / "P0_标注手册_v1.md"
    chain_doc = ROOT / "P0_标注链与金标冻结.md"
    check(
        "P0_标注链与金标冻结.md",
        chain_doc.is_file(),
        "the annotation-chain document is missing — data/annotation/ is git-ignored, so this is the only committed place that says what those artefacts are",
    )
    if chain_doc.is_file():
        check(
            "README.md",
            chain_doc.name in readme,
            f"{chain_doc.name} is not referenced from README.md",
        )
    if manual_path.is_file():
        manual = manual_path.read_text(encoding="utf-8")
        sheets_source = (ROOT / "src" / "context_benchmark" / "sheets.py").read_text(encoding="utf-8")
        rendered = re.search(r"格式：`([^`]+)`", sheets_source)
        if rendered:
            header = rendered.group(1)
            check(
                "P0_标注手册_v1.md",
                header in manual,
                f"the sheet renders the answer block as {header!r} but the manual does not document that format",
            )

    if problems:
        print("documentation consistency check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"documentation consistency check passed "
        f"(candidate_truth={truth_sha[:12]}…, tasks={tasks_sha[:12]}…, tests={tests_n})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
