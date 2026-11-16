# Contributing

Thanks for looking. This project is at P0 — there are no baseline scores yet, and the most valuable contributions right now are **annotation** and **criticism**, not features.

## Ground rules

Four of them, all about not fooling ourselves:

1. **Never present a self-check as a result.** `self-check` feeds the answer key back in and gets 1.0. If a PR adds a table of numbers, it must say where they came from.
2. **Data changes require a version bump.** Edit `candidate_truth.jsonl` and the hash changes; every previously reported number becomes unreproducible. Bump `dataset_version` in the manifest and state what changed. Do not edit published data silently — this is exactly the failure mode documented for LongMemEval in the design record.
3. **Every ratio reports its denominator.** A precision of 0.0 based on zero predictions is not a precision. Use the existing `ratio()` helper; it returns `null` for an empty denominator rather than a fake zero.
4. **Version everything.** A PR that adds data or scores must carry `benchmark_version`, `dataset_version`, a pinned model id and artifact hashes, via `submission.json` or `dataset.json`. An unversioned number cannot be compared with anything later.

## Getting set up

```bash
git clone https://github.com/<you>/context-induction-benchmark.git
cd context-induction-benchmark
export PYTHONPATH=src          # Windows: set PYTHONPATH=src
python -m unittest discover -s tests -v
```

No dependencies to install. If your change needs one, open an issue first — the zero-dependency constraint is intentional, so that reproduction never fails on a packaging problem.

## Workflow

Fork → branch → commit → pull request.

```bash
git checkout -b fix/r2-truncation-guard
# ... change ...
PYTHONPATH=src python -m unittest discover -s tests -v
git commit -m "fix: R2 guard must fail the task, not truncate it"
git push origin fix/r2-truncation-guard
```

Then open a PR against `main`. Describe **what you ran** and **what you saw**, not just what you changed. A PR that adds a scoring rule without a test will be asked for one.

## Submitting data or run results

Two package types. Both are checked by the same command and both are **append-only** — `data/pilot_v0` is frozen and must never be edited in place.

| Package | Manifest file | Produced by | Contains |
|---|---|---|---|
| Run result | `submission.json` | `run-induction --manifest <dir>` or `run-manifest` | `predictions.jsonl`, `calls.jsonl`, version stamps, hashes, usage and failure detail |
| Sample extension | `dataset.json` | `generate --output-dir <dir>` | a new generator-produced sample set, its seed, split, provenance and hashes |

**Run result:**

```bash
PYTHONPATH=src python -m context_benchmark run-induction \
  --runtime configs/runtime.mine.json --tasks data/pilot_v0/tasks.jsonl \
  --prompt prompts/induction_v1.txt \
  --output runs/preds.jsonl --logs runs/calls.jsonl \
  --manifest runs/submission --contributor <your-handle> --dimensions A C D
PYTHONPATH=src python -m context_benchmark validate-submission --dir runs/submission
```

**Sample extension** — copy `configs/generator.json` and change **both** `dataset_version` and `seed`:

```bash
PYTHONPATH=src python -m context_benchmark generate \
  --config configs/generator.mine.json \
  --output-dir data/contrib/<dataset_id> \
  --contributor <your-handle> --split public
PYTHONPATH=src python -m context_benchmark validate-submission --dir data/contrib/<dataset_id>
```

Open a PR with the whole directory. `--split` accepts `public`, `private`, `holdout`, `unspecified`; to run against an extension set, point `--tasks` at its `tasks.jsonl`.

The manifest must state: `benchmark_version` and `dimension_set_version` (from `configs/benchmark.json`), `dataset_version` and `split`, `evidence_grade` and `human_annotation_complete` (do **not** claim `human_gold` without adjudicated annotations), `model_id` + `pinned_version` (+ `weight_sha256` / `quant` / `sku` where applicable), prompt versions and hashes, `scorer_version`, token / latency / cost totals, failure counts, `dialogue_n = scored_n + failed_n`, contributor and license, and a SHA-256 for every artifact in the package. Real redacted logs additionally need `source_type: real_redacted` and a compliance note.

`validate-submission` re-hashes every declared file and exits 1 on any problem. It rejects unsupported benchmark versions, dimension sets that are not `six-axis-v1`, unknown or duplicated dimension ids, score declarations that omit the primary dimension A, unpinned models, mismatched denominators, fake human gold, missing compliance notes, missing artifacts and hash mismatches. CI runs the same check. Full contract: `schemas/submission.schema.json`; design rationale: design document §13.

If you are declaring scores, pass `--scores-declared`; the default is `false`, and the validator does not endorse contributor-declared numbers.

## What we are actively looking for

### 1. Human annotation of the calibration set (highest value)

This is the project's largest gap. Without it there is no κ and no human gold.

```bash
PYTHONPATH=src python -m context_benchmark make-sheets \
  --ids p0-001 p0-006 p0-016 p0-031 p0-040 \
  --out-dir data/annotation/calibration_mine \
  --annotator <your-initials>
```

Read `P0_标注手册_v1.md` (Chinese, 10 minutes), fill in the answer block at the bottom of each sheet, then:

```bash
PYTHONPATH=src python -m context_benchmark parse-sheets \
  --dir data/annotation/calibration_mine \
  --annotator <your-initials> \
  --output data/annotation/annotations_mine.jsonl \
  --report data/annotation/report_mine.json
```

Open a PR with the two output files. If the manual is unclear at any point, add `#疑问 <your question>` to the line — the parser collects those into the report, and they are the most useful thing you can send us.

**What we will do with it**: compute agreement against `P0_预检报告_AI对照.md` and against other annotators, and use every disagreement to revise the ontology. Your name goes in the contributors list unless you ask otherwise.

### 2. Independent reproduction

Run the audit and self-check on your machine and tell us whether you get the same hashes and the same 420/420. If you do not, that is a bug in reproducibility and we want to know.

### 3. Ontology criticism

Read section 2 and 5 of the annotation manual. If any of R1–R8 is ambiguous, or if you can construct a dialogue where two careful annotators would disagree, open an issue with the dialogue. Concrete counterexamples beat abstract objections.

### 4. Code

Good first issues: tokenizer-accurate budget accounting, matcher calibration tooling, the remaining steps of the eleven-step generator. Check the issue tracker before starting anything large.

### 5. Sample extensions

A new seed plus a new `dataset_version` produces an independent extension set under `data/contrib/<dataset_id>/`. Small, well-described sets are more useful than large ones right now: say what varies (scenario, conflict structure, injection position), which `split` it belongs to, and what failure mode it is meant to expose. See the submission format above.

## What we will not merge

- Changes that make a score look better without changing what is measured.
- New dimensions without an orthogonality check against the six registered axes (A–F). G (user identity / session boundary) is reserved and inactive until P2.
- A submission without a version stamp, or one whose artifact hashes do not match.
- An "extension" set whose `tasks.jsonl` is byte-identical to the frozen pilot — changing the directory name is not a new sample.
- Anything that calls a paid API inside the default test suite.
- Data edits without a version bump.

## Style

- Python: standard library only, type hints, no formatting religion.
- Prose: state the claim, then the evidence, then the limitation.
- Commit messages: `type: what changed`. Types in use: `feat`, `fix`, `data`, `docs`, `test`, `refactor`.

## Questions

Open an issue. If it is about the design rather than the code, the design document is the place to start — it records what was tried and rejected, including four rounds of narrowing the contribution claim.
