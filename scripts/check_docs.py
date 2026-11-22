"""Single-source-of-truth check for the numbers quoted in the documentation.

The audit found that the dataset SHA-256 and the test count were quoted
differently in README.md, P0_标注规范与试点方案.md and data/pilot_v0/manifest.json.
This script recomputes every quoted value and fails when a document drifts.

Two properties matter more than the individual gates:

* **Every gate is a function that returns its own problem list.** A gate that
  only exists inside ``main()`` cannot be tested, and an untested gate is a
  gate that silently stops working. ``tests/test_docs_gates.py`` exercises
  each one against a deliberately broken input.
* **A gate that cannot fail is worse than no gate.** An earlier version of the
  test-count regex only knew the English word "tests", so it never matched a
  Chinese sentence and reported nothing at all. The pattern now covers both
  languages, and the test suite asserts that it fires.

Run: python scripts/check_docs.py
Exit code 1 on any inconsistency.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_TRUTH = ROOT / "data" / "pilot_v0" / "candidate_truth.jsonl"
TASKS = ROOT / "data" / "pilot_v0" / "tasks.jsonl"
README = ROOT / "README.md"
P0_DOC = ROOT / "P0_标注规范与试点方案.md"
MANIFEST = ROOT / "data" / "pilot_v0" / "manifest.json"
CALIBRATION = ROOT / "configs" / "calibration.json"
BENCHMARK_CONFIG = ROOT / "configs" / "benchmark.json"
SELF_CHECK = ROOT / "data" / "pilot_v0" / "self_check.json"
CONTRIB_TRUTH = ROOT / "data" / "contrib" / "pilot-v1-randomized" / "candidate_truth.jsonl"
MANUAL = ROOT / "P0_标注手册_v1.md"
CHAIN_DOC = ROOT / "P0_标注链与金标冻结.md"
SCORING_SOURCE = ROOT / "src" / "context_benchmark" / "scoring.py"
CLI_SOURCE = ROOT / "src" / "context_benchmark" / "cli.py"
SHEETS_SOURCE = ROOT / "src" / "context_benchmark" / "sheets.py"

# Every document that ships with the repository and quotes numbers a reader
# could copy. The drift gates scan *all* of them: an earlier version only
# looked at README.md and the P0 plan, which is why a stale hash and a stale
# test count survived in three other files for days.
DOC_FILES = (
    "README.md",
    "P0_标注规范与试点方案.md",
    "P0_标注手册_v1.md",
    "P0_预检报告_AI对照.md",
    "P0_标注链与金标冻结.md",
    "P1_能力评测改造计划_2026-11-22.md",
    "上下文归纳评测框架_设计文档.md",
    "任务交接_2026-11-21.md",
    "参考项目深度研读台账.md",
)

# "197 个测试全绿" / "47 项测试通过" / "197 tests". The Chinese word 测试 and
# the Chinese counter 项/个 are both needed: the previous pattern only knew
# "tests", so a Chinese sentence never matched and the gate was dead code.
TEST_COUNT_RE = re.compile(
    r"(\d{2,4})\s*(?:\+|个)?\s*(?:项|个)?\s*(?:tests?(?![A-Za-z])|测试|单元测试)"
)

# A quoted test count that is explicitly historical ("原 180 + 新增 17") is a
# record of what a past commit contained, not a claim about the current suite.
# "前" alone is deliberately NOT a marker: it would exempt "当前 197 个测试".
HISTORICAL_COUNT_RE = re.compile(r"(?:原|新增|此前|之前)\s*[0-9]*\s*$")

CALIBRATION_THRESHOLD_KEYS = {
    "text_min", "anchor_coverage_min", "evidence_iou_min", "slot_match_required",
}
CALIBRATION_PROVENANCE_KEYS = (
    "calibration_set", "calibration_set_sha256", "adjudicator",
    "fitted_at", "sample_n", "sensitivity", "specificity", "split",
)


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


def load_docs(root: Path | None = None) -> tuple[dict[str, str], list[str]]:
    base = root or ROOT
    docs: dict[str, str] = {}
    problems: list[str] = []
    for name in DOC_FILES:
        path = base / name
        if path.is_file():
            docs[name] = path.read_text(encoding="utf-8")
        else:
            problems.append(f"{name}: listed in DOC_FILES but the file does not exist")
    return docs, problems


# ---------------------------------------------------------------------------
# Gate 1 — dataset hashes quoted in the documents must match the files.
# ---------------------------------------------------------------------------

def gate_1_hashes(docs: dict[str, str], truth_sha: str, tasks_sha: str) -> list[str]:
    problems: list[str] = []
    for name, text in docs.items():
        for quoted in set(re.findall(r"\b[0-9a-f]{8,64}\b", text)):
            if len(quoted) < 40 or quoted in {truth_sha, tasks_sha}:
                continue
            for context in re.finditer(rf"[^\n]*{quoted}[^\n]*", text):
                line = context.group(0)
                if re.search(r"(sha|hash|哈希|SHA-256)", line, re.IGNORECASE):
                    problems.append(
                        f"{name}: quoted hash {quoted[:12]}… does not match any current artifact "
                        f"(candidate_truth={truth_sha[:12]}…, tasks={tasks_sha[:12]}…) — line: {line.strip()[:90]}"
                    )
    return problems


# ---------------------------------------------------------------------------
# Gate 2 — the frozen tasks hash in configs/benchmark.json must be current.
# ---------------------------------------------------------------------------

def gate_2_frozen_tasks_hash(tasks_sha: str, config_text: str | None = None) -> list[str]:
    text = config_text if config_text is not None else BENCHMARK_CONFIG.read_text(encoding="utf-8")
    declared = re.search(r'"frozen_tasks_sha256":\s*"([0-9a-f]{64})"', text)
    if not declared:
        return ["configs/benchmark.json: frozen_tasks_sha256 is missing"]
    if declared.group(1) != tasks_sha:
        return [
            f"configs/benchmark.json: frozen_tasks_sha256={declared.group(1)[:12]}… "
            f"but tasks.jsonl is {tasks_sha[:12]}…"
        ]
    return []


# ---------------------------------------------------------------------------
# Gate 3 — every quoted test count must match the real number of test methods.
# ---------------------------------------------------------------------------

def gate_3_test_counts(docs: dict[str, str], tests_n: int) -> list[str]:
    problems: list[str] = []
    for name, text in docs.items():
        for match in TEST_COUNT_RE.finditer(text):
            value = int(match.group(1))
            prefix = text[max(0, match.start() - 16):match.start()]
            if HISTORICAL_COUNT_RE.search(prefix):
                continue
            if value > 10000 or value == tests_n:
                continue
            line_no = text[:match.start()].count("\n") + 1
            problems.append(
                f"{name}:{line_no}: quotes {value} tests but the suite currently has {tests_n} test methods"
            )
    return problems


# ---------------------------------------------------------------------------
# Gate 4 — the manifest must record the dataset version the documents claim.
# ---------------------------------------------------------------------------

def gate_4_manifest_version(docs: dict[str, str], manifest_text: str) -> list[str]:
    problems: list[str] = []
    version = re.search(r'"dataset_version":\s*"([^"]+)"', manifest_text)
    if not version:
        return ["data/pilot_v0/manifest.json: dataset_version missing"]
    readme = docs.get("README.md", "")
    if version.group(1) not in readme:
        problems.append(
            f"README.md: dataset version {version.group(1)} is not mentioned in README.md"
        )
    return problems


# ---------------------------------------------------------------------------
# Gate 5 — prompt file names must match the versions they declare.
# ---------------------------------------------------------------------------

def gate_5_prompt_versions(prompts_dir: Path | None = None) -> list[str]:
    base = prompts_dir or (ROOT / "prompts")
    problems: list[str] = []
    for prompt in sorted(base.glob("*.txt")):
        first = prompt.read_text(encoding="utf-8").splitlines()[0]
        declared_version = re.search(r"PROMPT_VERSION:\s*(\S+)", first)
        if not declared_version:
            problems.append(f"{prompt.name}: no PROMPT_VERSION header")
            continue
        suffix = declared_version.group(1).rsplit("-", 1)[-1]
        stem_version = re.search(r"_(v\d+)\.txt$", prompt.name)
        if stem_version and stem_version.group(1) != suffix:
            problems.append(
                f"{prompt.name}: declares PROMPT_VERSION {declared_version.group(1)} "
                f"but the file name says {stem_version.group(1)}"
            )
    return problems


# ---------------------------------------------------------------------------
# Gate 6 — implementation_status must mirror the README and point at real files.
# ---------------------------------------------------------------------------

def gate_6_implementation_status(docs: dict[str, str], config: dict) -> list[str]:
    problems: list[str] = []
    status = config.get("implementation_status") or {}
    entries = status.get("entries") or []
    readme = docs.get("README.md", "")
    if not entries:
        problems.append(
            "configs/benchmark.json: implementation_status.entries is missing — "
            "the README claims this table is machine-readable"
        )
    for entry in entries:
        promise = entry.get("promise", "")
        state = entry.get("state", "")
        if state not in {"implemented", "partial", "not_implemented"}:
            problems.append(
                f"configs/benchmark.json: implementation_status[{entry.get('id')}].state={state!r} is not a legal state"
            )
        if promise and promise not in readme:
            problems.append(
                f"configs/benchmark.json: implementation_status[{entry.get('id')}] promise "
                f"{promise[:40]!r} is not in the README table"
            )
        for evidence_path in entry.get("evidence", []):
            if not (ROOT / evidence_path).exists():
                problems.append(
                    f"configs/benchmark.json[{entry.get('id')}]: evidence path {evidence_path!r} does not exist"
                )
        if state == "implemented" and not entry.get("evidence"):
            problems.append(
                f"configs/benchmark.json[{entry.get('id')}]: state 'implemented' but no evidence is listed"
            )
    return problems


# ---------------------------------------------------------------------------
# Gate 7 — the randomised extension set must carry the false-trigger control.
# ---------------------------------------------------------------------------

def gate_7_false_trigger_probes(truth_path: Path | None = None) -> list[str]:
    path = truth_path or CONTRIB_TRUTH
    if not path.is_file():
        return []
    related_false = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for probe in json.loads(line).get("probes", []) or []:
            if probe.get("related") is False:
                related_false += 1
    if related_false == 0:
        return [
            "data/contrib/pilot-v1-randomized: the randomised extension set has no related=false probes, "
            "so dimension B still has no false-trigger estimate"
        ]
    return []


# ---------------------------------------------------------------------------
# Gate 8 — the committed self-check must quote the versions the code declares.
# ---------------------------------------------------------------------------

def gate_8_self_check_versions(
    artifact: dict, scoring_source: str, cli_source: str,
) -> list[str]:
    problems: list[str] = []
    matcher = re.search(r'MATCHER_VERSION\s*=\s*"([^"]+)"', scoring_source)
    scorer = re.search(r'SCORER_VERSION\s*=\s*"([^"]+)"', cli_source)
    if matcher:
        if artifact.get("matcher_version") != matcher.group(1):
            problems.append(
                f"data/pilot_v0/self_check.json: matcher_version={artifact.get('matcher_version')!r} "
                f"but scoring.py declares {matcher.group(1)!r}"
            )
    if scorer:
        if artifact.get("scorer_version") != scorer.group(1):
            problems.append(
                f"data/pilot_v0/self_check.json: scorer_version={artifact.get('scorer_version')!r} "
                f"but cli.py declares {scorer.group(1)!r}"
            )
    return problems


# ---------------------------------------------------------------------------
# Gate 9 — the annotator manual and the rendered sheet must agree.
# ---------------------------------------------------------------------------

def gate_9_manual_format(docs: dict[str, str], sheets_source: str) -> list[str]:
    problems: list[str] = []
    if not CHAIN_DOC.is_file():
        problems.append(
            "P0_标注链与金标冻结.md: the annotation-chain document is missing — data/annotation/ is "
            "git-ignored, so this is the only committed place that says what those artefacts are"
        )
    else:
        readme = docs.get("README.md", "")
        if CHAIN_DOC.name not in readme:
            problems.append(f"README.md: {CHAIN_DOC.name} is not referenced from README.md")
    if MANUAL.is_file():
        manual = MANUAL.read_text(encoding="utf-8")
        rendered = re.search(r"格式：`([^`]+)`", sheets_source)
        if rendered and rendered.group(1) not in manual:
            problems.append(
                f"P0_标注手册_v1.md: the sheet renders the answer block as {rendered.group(1)!r} "
                "but the manual does not document that format"
            )
    return problems


# ---------------------------------------------------------------------------
# Gate 10 — scoring thresholds must carry their provenance.
# ---------------------------------------------------------------------------

def gate_10_calibration_provenance(
    calibration: dict, scoring_source: str, root: Path | None = None,
) -> list[str]:
    base = root or ROOT
    problems: list[str] = []
    state = calibration.get("state")
    if state not in {"uncalibrated", "calibrated"}:
        problems.append(f"configs/calibration.json: state={state!r} is not a legal state")
    thresholds = calibration.get("thresholds") or {}
    provenance = calibration.get("provenance") or {}
    if set(thresholds) != CALIBRATION_THRESHOLD_KEYS:
        problems.append(
            f"configs/calibration.json: thresholds keys are {sorted(thresholds)}, "
            f"expected {sorted(CALIBRATION_THRESHOLD_KEYS)}"
        )
    code_text_min = re.search(r"TEXT_MATCH_MIN(?:_DEFAULT)?\s*=\s*([0-9.]+)", scoring_source)
    code_anchor_min = re.search(r"ANCHOR_COVERAGE_MIN(?:_DEFAULT)?\s*=\s*([0-9.]+)", scoring_source)
    if state == "uncalibrated":
        # The config must not drift away from the gates the scorer actually
        # uses while nobody has calibrated anything.
        if code_text_min:
            try:
                same = float(thresholds.get("text_min", -1)) == float(code_text_min.group(1))
            except (TypeError, ValueError):
                same = False
            if not same:
                problems.append(
                    f"configs/calibration.json: text_min={thresholds.get('text_min')} "
                    f"but scoring.py uses {code_text_min.group(1)}"
                )
        if code_anchor_min:
            try:
                same = float(thresholds.get("anchor_coverage_min", -1)) == float(code_anchor_min.group(1))
            except (TypeError, ValueError):
                same = False
            if not same:
                problems.append(
                    f"configs/calibration.json: anchor_coverage_min={thresholds.get('anchor_coverage_min')} "
                    f"but scoring.py uses {code_anchor_min.group(1)}"
                )
    else:
        # A calibrated threshold the scorer never reads is a decoration: the
        # config would claim a number that no score was produced with.
        if "calibration.json" not in scoring_source:
            problems.append(
                "configs/calibration.json: state is 'calibrated' but scoring.py never reads "
                "configs/calibration.json — the calibrated thresholds would not be in force"
            )
    if state == "uncalibrated":
        filled = [key for key, value in provenance.items() if value not in (None, [], "", 0)]
        if filled:
            problems.append(
                f"configs/calibration.json: state is 'uncalibrated' but provenance carries values for {filled}"
            )
        if thresholds.get("evidence_iou_min") is not None or thresholds.get("slot_match_required") is not False:
            problems.append(
                "configs/calibration.json: uncalibrated state must not claim the P1 thresholds "
                "(evidence_iou_min / slot_match_required)"
            )
    else:
        for key in CALIBRATION_PROVENANCE_KEYS:
            value = provenance.get(key)
            if value in (None, "", [], 0):
                problems.append(
                    f"configs/calibration.json: state is 'calibrated' but provenance.{key} is empty"
                )
        if not provenance.get("annotators"):
            problems.append(
                "configs/calibration.json: state is 'calibrated' but provenance.annotators is empty"
            )
        sample = provenance.get("calibration_set")
        if sample:
            sample_path = base / sample
            if not sample_path.is_file():
                problems.append(
                    f"configs/calibration.json: the calibration set {sample!r} does not exist"
                )
            elif sha256_file(sample_path) != provenance.get("calibration_set_sha256"):
                problems.append(
                    "configs/calibration.json: the calibration set hash does not match "
                    "provenance.calibration_set_sha256"
                )
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    truth_sha = sha256_file(CANDIDATE_TRUTH)
    tasks_sha = sha256_file(TASKS)
    tests_n = count_tests()

    docs, problems = load_docs()
    problems += gate_1_hashes(docs, truth_sha, tasks_sha)
    problems += gate_2_frozen_tasks_hash(tasks_sha)
    problems += gate_3_test_counts(docs, tests_n)
    problems += gate_4_manifest_version(docs, MANIFEST.read_text(encoding="utf-8"))
    problems += gate_5_prompt_versions()
    problems += gate_6_implementation_status(
        docs, json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    )
    problems += gate_7_false_trigger_probes()
    if SELF_CHECK.is_file():
        problems += gate_8_self_check_versions(
            json.loads(SELF_CHECK.read_text(encoding="utf-8")),
            SCORING_SOURCE.read_text(encoding="utf-8"),
            CLI_SOURCE.read_text(encoding="utf-8"),
        )
    problems += gate_9_manual_format(docs, SHEETS_SOURCE.read_text(encoding="utf-8"))
    if not CALIBRATION.is_file():
        problems.append("configs/calibration.json: the calibration config is missing")
    else:
        problems += gate_10_calibration_provenance(
            json.loads(CALIBRATION.read_text(encoding="utf-8")),
            SCORING_SOURCE.read_text(encoding="utf-8"),
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
