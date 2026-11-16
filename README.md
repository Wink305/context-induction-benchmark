# Context Induction Benchmark (CIB)

A diagnostic benchmark for measuring whether a language model can **decide what is worth remembering** — which utterances in a long conversation are durable constraints, which are one-off requests, and which are already dead.

> **Project status: P0 — engineering ready, no scores yet.**
>
> - **No baseline scores exist.** This repository has never called a real model API. Every number produced so far comes from a mock provider or a scorer self-check.
> - **The labels are not human-annotated gold.** `candidate_truth.jsonl` is produced by a deterministic generator and validated by a rule audit (`audit.py`). Its evidence grade is `generator_truth_rule_audited`, **not** `human gold`. No inter-annotator agreement (κ) has been measured.
> - **The data is synthetic and template-based** (generator v0). It has not been through the eleven-step generation pipeline described in the design document.
> - **No ablation has been run**, so construct validity — that this measures induction rather than long-context retrieval — is argued, not demonstrated.
>
> If you need a working memory benchmark today, use [LongMemEval](https://github.com/xiaowu0162/LongMemEval) or [MemoryAgentBench](https://arxiv.org/abs/2507.05257). This project is an attempt to cover a gap those leave, not a replacement.

---

## What it measures

Existing memory benchmarks evaluate whether a system can **store and retrieve**. This one evaluates the upstream step: whether a model can **tell what deserves to be stored at all**.

Six axes, in three roles: **one primary dimension**, three rankable supporting dimensions, two attribution-only axes.

| | Role | Dimension | What it asks |
|---|---|---|---|
| **A** | **Primary** | Constraint discrimination | Can it separate a durable constraint from a one-off request? Human κ and the project's stop condition hang on this one. |
| **B** | Supporting | Association & transfer safety | When a distilled point is triggered in a new situation, is the trigger correct — and what does over-generalization cost? |
| **C** | Supporting | Conflict & staleness | When an old rule and a new rule collide, does it know which one is live? |
| **D** | Supporting | Hallucination resistance & abstention | Does it invent constraints the user never authorised, and can it output an empty set when there is nothing to remember? |
| **E** *(was R1)* | Attribution | Decay curve | At which turn does recall of early constraints start to fall? (Kaplan–Meier, reported as survival) |
| **F** *(was R2)* | Attribution | Durability baseline | With compression and truncation forbidden, how far can the model get before it collapses? |

A/B/C/D are reported as a **profile, never a weighted total**. E/F are attribution-only: they never enter the ranking. D was split out of A because `must-include` and `must-exclude` have different denominators and must not be folded into one accuracy number (design §5.5). E/F are the former R1/R2 promoted to named dimensions — this v0.4 change is a *re-registration*, not three new constructs, and orthogonality between the rankable axes still has to be established by the P1 correlation matrix. The machine-readable registry is `src/context_benchmark/dimensions.py` (`python -m context_benchmark dimensions`).

## Quick start

No API key, no third-party dependencies. Python 3.10+.

```bash
git clone https://github.com/Wink305/context-induction-benchmark.git
cd context-induction-benchmark
export PYTHONPATH=src          # Windows: set PYTHONPATH=src

python -m unittest discover -s tests -v          # 80 tests
python -m context_benchmark dimensions           # print the six-axis registry
python -m context_benchmark generate --config configs/generator.json
python -m context_benchmark audit    --candidate data/pilot_v0/candidate_truth.jsonl --output data/pilot_v0/audit.json
python -m context_benchmark self-check --candidate data/pilot_v0/candidate_truth.jsonl
```

`generate` is deterministic: the same seed reproduces byte-identical output (`candidate_truth.jsonl` SHA-256 `57503739…`).

`self-check` feeds the candidate truth back in as if it were a model prediction. It should report 1.0 on every metric. **That is a scorer sanity check, not a model result.**

## Repository layout

```
configs/           generator, runtime, model-role and benchmark version configuration
prompts/           fixed induction and executor prompts (versioned)
schemas/           JSON Schema contracts: generator output, model output, submissions
src/context_benchmark/
  dimensions.py    six-axis registry, primary dimension, dimension-set validation
  manifest.py      version stamping, artifact hashing, submission validation
  ontology.py      single source of truth for labels and conditions
  generator.py     deterministic dialogue generator (v0), append-only extensions
  validation.py    schema plus field-level semantic validation
  scoring.py       dual-list, FAMA, FMR, empty-output, counterfactual arms
  agreement.py     Cohen's κ, clustered bootstrap CI, Gwet's AC1
  adjudication.py  third-reviewer decisions; frozen gold cannot be overwritten
  harness.py       two-stage harness, four background ablations, three arms
  providers.py     mock and OpenAI-compatible providers (real calls fail closed)
  survival.py      Kaplan–Meier, REC/ROB
  audit.py         rule audit of the candidate truth
  sheets.py        annotation sheet render / parse
tests/             80 unit and adversarial tests
data/pilot_v0/     frozen P0 candidate data (see below)
```

## Data

`data/pilot_v0/` contains 40 dialogues (26–40 turns each), 420 candidate annotation units, 10 planted pseudo-constraints and 70 behaviour probes.

| File | What it is | Safe to use as gold? |
|---|---|---|
| `tasks.jsonl` | The model-facing input | — |
| `candidate_truth.jsonl` | Generator-produced labels | **No** — see status note |
| `annotation_worksheet.jsonl` | Answer-free sheets for human annotation | — |
| `audit.json` | Rule-audit result (420/420 clean) | — |
| `coverage.json`, `manifest.json` | Coverage checks and file hashes | — |
| `calibration_ids.json` | The 5-dialogue calibration subset | — |

The five calibration dialogues each plant a different trap: a standard segment, a same-structure segment in another domain, a double-withdrawal segment, a pseudo-constraint segment, and an empty segment where the correct answer is to output nothing.

## Using your own model

Real API calls are disabled by default (`allow_api_calls: false`) and will fail closed until a verified model id and pinned version are supplied. This is deliberate: aliases drift, and an unpinned model makes a benchmark unreproducible.

```bash
# 1. copy the runtime config and fill in a *pinned* model id, base_url, api_key_env
# 2. set allow_api_calls to true
python -m context_benchmark run-induction \
  --runtime configs/runtime.mine.json \
  --tasks data/pilot_v0/tasks.jsonl \
  --prompt prompts/induction_v1.txt \
  --output runs/preds.jsonl \
  --logs runs/calls.jsonl
```

Every request, response, token count, latency and cost is written to the call log. Failed calls, schema violations and semantic errors are **counted as failures**, never silently dropped.

## Versioning and submissions

`configs/benchmark.json` is the single source of truth (`benchmark_version = 0.4.0`, `dimension_set_version = six-axis-v1`). A score is meaningless without the exact benchmark, dataset, prompts, schema, scorer and model that produced it, so every run can emit a self-contained submission directory:

```bash
PYTHONPATH=src python -m context_benchmark run-induction \
  --runtime configs/runtime.mine.json \
  --tasks data/pilot_v0/tasks.jsonl \
  --prompt prompts/induction_v1.txt \
  --output runs/preds.jsonl --logs runs/calls.jsonl \
  --manifest runs/submission --contributor <your-handle> --dimensions A C D

PYTHONPATH=src python -m context_benchmark validate-submission --dir runs/submission
```

`runs/submission/` contains `submission.json`, `predictions.jsonl` and `calls.jsonl`. The manifest stamps:

| | Recorded |
|---|---|
| Versions | `benchmark_version`, `dimension_set_version`, `primary_dimension`, `dimensions_reported`, `scores_declared`, schema and prompt versions, `scorer_version` (`scoring.py@<sha12>`) |
| Dataset | `dataset_version`, `split` (public/private/holdout), `evidence_grade`, `human_annotation_complete`, `gold_version`, seed, generator config and dataset manifest hashes |
| Model | `model_id`, `pinned_version`, `weight_sha256`, `quant`, `sku`, provider, temperature, seed, `repetition_index` |
| Run detail | token counts, latency, cost, provider errors, invalid outputs, `dialogue_n = scored_n + failed_n` |
| Provenance | contributor, license, compliance note, SHA-256 of every artifact in the package and of the repository files that produced the score |

The validator re-hashes every declared artifact and rejects unsupported versions, unpinned models, mismatched denominators, fake `human_gold`, missing compliance notes for real data, and score declarations that omit the primary dimension. It exits 1 on failure and runs in CI. Contract: `schemas/submission.schema.json`; implementation: `src/context_benchmark/manifest.py`.

### Extending the sample set

`data/pilot_v0` is **frozen** — never edit it in place. New samples are append-only, in their own directory with their own `dataset.json`:

```bash
# copy configs/generator.json, then change BOTH dataset_version and seed
PYTHONPATH=src python -m context_benchmark generate \
  --config configs/generator.mine.json \
  --output-dir data/contrib/<dataset_id> \
  --contributor <your-handle> --split public

PYTHONPATH=src python -m context_benchmark validate-submission --dir data/contrib/<dataset_id>
```

Two checks are enforced: an extension cannot reuse `pilot-v0-candidate`, and a `tasks.jsonl` that is byte-identical to the frozen pilot is refused — renaming a directory is not a new sample. Real redacted logs require `source_type: real_redacted` plus a compliance note. See [CONTRIBUTING.md](CONTRIBUTING.md) and design document §13.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: fork, branch, run the tests, open a PR. Any change to data or scoring thresholds needs a version bump — no silent edits.

The most useful contribution right now is **not code**:

- **Human annotation of the calibration set.** 5 dialogues, ~60 units, 40–60 minutes. This is the single biggest gap: without it there is no κ and no human gold.
- **Independent reproduction** of the audit result on your machine.
- **Criticism of the ontology.** If R1–R8 rules are ambiguous to you, that is a finding, not a nuisance — open an issue.

## Design documents

The repository is documentation-first; the design record is in Chinese.

| File | Contents |
|---|---|
| `上下文归纳评测框架_设计文档.md` | Full framework, dimension operationalization, citation errata, evidence grading |
| `P0_标注规范与试点方案.md` | Decisions, ontology, annotation protocol, cost model, roadmap, stop conditions |
| `参考项目深度研读台账.md` | Close reading of 19 primary papers, with what was borrowed and what was corrected |
| `P0_标注手册_v1.md` | Annotator manual (v1.1) |
| `P0_预检报告_AI对照.md` | What an AI cross-check found, including two ontology errors |

## Known limitations

1. Synthetic dialogues are template-generated; lexical diversity is low and the implicit-constraint cues may be more regular than real user behaviour.
2. The text matcher is **uncalibrated**. It is usable for engineering checks, not for reporting model scores.
3. `raw_turns` is a `first_recent_v0` baseline, not a reproduction of the selection policy in arXiv:2609.34227.
4. Only 40 dialogues. Anything inferred from them is a pilot signal.
5. No external reproduction, no pre-registration, no peer review.
6. The six-axis set is a **structural re-registration**, not six independently validated constructs. D was split from A and E/F renamed from R1/R2; their orthogonality has not been measured because the P1 correlation matrix has not been run. A pairwise r > 0.8 would merge them back.

## License

MIT. See [LICENSE](LICENSE).
