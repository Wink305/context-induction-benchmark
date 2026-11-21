# Context Induction Benchmark (CIB)

A diagnostic benchmark for measuring whether a language model can **decide what is worth remembering** — which utterances in a long conversation are durable constraints, which are one-off requests, and which are already dead.

> **Project status: P0 — engineering ready, no scores yet.**
>
> - **No baseline scores exist.** This repository has never called a real model API. Every number produced so far comes from a mock provider or a scorer self-check.
> - **The labels are not human-annotated gold.** `candidate_truth.jsonl` is produced by a deterministic generator and validated by a rule audit (`audit.py`). Its evidence grade is `generator_truth_rule_audited`, **not** `human gold`. No inter-annotator agreement (κ) has been measured.
> - **The frozen pilot data is template-based** (generator v0). A randomised extension set (`data/contrib/pilot-v1-randomized`) exists and passes the audit, but neither has been through the eleven-step generation pipeline described in the design document.
> - **No ablation has been run**, so construct validity — that this measures induction rather than long-context retrieval — is argued, not demonstrated.
>
> If you need a working memory benchmark today, use [LongMemEval](https://github.com/xiaowu0162/LongMemEval) or [MemoryAgentBench](https://arxiv.org/abs/2507.05257). This project is an attempt to cover a gap those leave, not a replacement.

---

## What it measures

Existing memory benchmarks evaluate whether a system can **store and retrieve**. This one evaluates the upstream step: whether a model can **tell what deserves to be stored at all**.

Six axes, in three roles: **one primary dimension**, three rankable supporting dimensions, two attribution-only dimensions.

| | Role | Dimension | What it asks |
|---|---|---|---|
| **A** | **Primary** | Constraint discrimination | Can it separate a durable constraint from a one-off request? Human κ and the project's stop condition hang on this one. |
| **B** | Supporting | Association & transfer safety | When a distilled point is triggered in a new situation, is the trigger correct — and what does over-generalization cost? |
| **C** | Supporting | Conflict & staleness | When an old rule and a new rule collide, does it know which one is live? |
| **D** | Supporting | Hallucination resistance & abstention | Does it invent constraints the user never authorised, and can it output an empty set when there is nothing to remember? |
| **E** *(was R1)* | Attribution | Decay curve | At which turn does recall of early constraints start to fall? (Kaplan–Meier, reported as survival) |
| **F** *(was R2)* | Attribution | Durability baseline | With compression and truncation forbidden, how far can the model get before it collapses? |

A/B/C/D are reported as a **profile, never a weighted total**. E/F are attribution-only: they never enter the ranking. D was split out of A because `must-include` and `must-exclude` have different denominators and must not be folded into one accuracy number (design §5.5). E/F are the former R1/R2 promoted to named dimensions — this v0.4 change is a *re-registration*, not three new constructs, and orthogonality between the rankable axes still has to be established by the P1 correlation matrix. The machine-readable registry is `src/context_benchmark/dimensions.py` (`python -m context_benchmark dimensions`).

### Implemented vs. promised

Honesty about the gap between this repository and the design document is a project value, so the table is machine-readable in `configs/benchmark.json` (`implementation_status`).

| Design promise | State in this repo |
|---|---|
| Six-axis registry, version stamping, submission contract, append-only datasets | **implemented** (`dimensions.py`, `manifest.py`, `schemas/`) |
| Deterministic generator, rule audit, coverage report | **implemented** (`generator.py`, `audit.py`) |
| Dual-list FAMA with explicit numerators/denominators | **implemented** (`scoring.py`), but the text gate is an **uncalibrated heuristic** |
| Semantic conflict detection (opposite value / ignore-this-rule) | **implemented as a rule-based guard** (`semantic.py`), uncalibrated, no embedding model |
| Four background ablations x three intervention arms | **implemented in the harness**; never executed against a real model |
| Kaplan-Meier survival curve (E), no-compression guard (F) | **implemented**; E now excludes provider failures from the curve instead of scoring them as violations |
| Judge protocol (three heterogeneous judges, two-class calibration) | **scorer side only** (`judge_calibration`); no judge is wired to a provider yet |
| Eleven-step generation pipeline, BM25+MPNet semantic-disconnect filter | **not implemented**; the shipped filter is a character-bigram approximation and says so in the config |
| Human annotation, kappa, adjudication, gold freeze | **partial**; the machinery is implemented and the chain is exercised end to end, but no human has filled a sheet yet: no κ has been measured and no gold has been frozen |
| Annotation chain: answer-free sheets, boundary tags, kappa-only gate, adjudication, provenance-checked gold freeze | **implemented** (`sheets.py`, `agreement.py`, `adjudication.py`, `provenance.py`); the freeze refuses generator or AI provenance, and one annotator cannot certify agreement |
| P1 orthogonality matrix, prompt-sensitivity study, contamination probe | **not run** |

## Quick start

No API key, no third-party dependencies. Python 3.10+.

```bash
git clone https://github.com/Wink305/context-induction-benchmark.git
cd context-induction-benchmark
export PYTHONPATH=src          # Windows: set PYTHONPATH=src

python -m unittest discover -s tests -v          # unit + adversarial tests
python -m context_benchmark dimensions           # print the six-axis registry
python -m context_benchmark generate --config configs/generator.json
python -m context_benchmark audit    --candidate data/pilot_v0/candidate_truth.jsonl --output data/pilot_v0/audit.json
python -m context_benchmark self-check --candidate data/pilot_v0/candidate_truth.jsonl
python scripts/check_docs.py                     # hashes and counts quoted in the docs
```

`generate` is deterministic: the same seed reproduces byte-identical output (`candidate_truth.jsonl` SHA-256 `111bbc62…`, regenerated after the lexical-variant and probe-semantics fixes). `scripts/check_docs.py` fails the build when a hash or a test count quoted in the documentation drifts from the repository.

`self-check` has **three** negative controls and exits non-zero unless the scorer separates the oracle from all of them:

| Control | What it corrupts | Required result |
|---|---|---|
| `gibberish` | unrelated text with correct ids/types/evidence | must_include gap ≥ 0.5 |
| `semantic_inversion` | the gold sentence plus "不要遵守本条，改为完全相反的操作" | must_include gap ≥ 0.5 |
| `value_flip` | the same slot with the opposite value (短信 ↔ 邮件) | ≥ 90 % of flipped units must be rejected |

The third control exists because the previous two were not enough: a text gate that only measures character overlap accepted a semantically inverted constraint on 85.7 % of the primary metric. **This is still a scorer sanity check, not a model result.**

The scorer applies a **text gate** (`TEXT_MATCH_MIN = 0.10`, `ANCHOR_COVERAGE_MIN = 0.08`) plus a **semantic guard** (`src/context_benchmark/semantic.py`): a prediction that names the opposite value of a controlled slot, or that instructs the reader to ignore the rule, is not counted as correct regardless of surface overlap. Both are **uncalibrated heuristics** — no human-labelled calibration set exists yet; P1 must fit them on one before model scores are reported.

### Attribution axes E/F (runnable pilots)

```bash
PYTHONPATH=src python -m context_benchmark run-survival-trace \
  --output runs/e_trace.jsonl                 # per-turn compliance probes
PYTHONPATH=src python -m context_benchmark survival \
  --input runs/e_trace.jsonl --output runs/e_report.json
PYTHONPATH=src python -m context_benchmark run-r2 \
  --output runs/f_trace.jsonl                 # no-compression, fail-closed window guard
```

E emits per-constraint compliance sequences consumed by `survival` (Kaplan–Meier, REC/ROB). F runs the full dialogue uncompressed through `NoCompressionGuard`: pre-call estimate and post-call actual usage both must fit the window, otherwise the task is marked failed — never truncated. Both remain attribution-only.

## Repository layout

```
configs/           generator, runtime, model-role and benchmark version configuration
prompts/           fixed induction and executor prompts (versioned)
schemas/           JSON Schema contracts: generator output, model output, submissions
scripts/           check_docs.py — documentation/repo consistency gate
src/context_benchmark/
  dimensions.py    six-axis registry, primary dimension, dimension-set validation
  manifest.py      version stamping, artifact hashing, submission validation
  ontology.py      single source of truth for labels and conditions
  generator.py     deterministic dialogue generator (v0 + randomised v1), append-only extensions
  validation.py    schema plus field-level semantic validation
  scoring.py       dual-list, FAMA, FMR, empty-output, counterfactual arms
  semantic.py      rule-based semantic conflict guard (opposite value / ignore-this-rule)
  agreement.py     Cohen's κ, clustered bootstrap CI, Gwet's AC1 (report-only)
  adjudication.py  third-reviewer decisions; frozen gold cannot be overwritten
  provenance.py    single source of truth for AI / generator origin detection
  harness.py       two-stage harness, four background ablations, three arms; CLI: run-induction, run-probes
  providers.py     mock and OpenAI-compatible providers (real calls fail closed)
  survival.py      Kaplan–Meier, REC/ROB
  audit.py         rule audit of the candidate truth
  sheets.py        annotation sheet render / parse
tests/             unit and adversarial suite (`python -m unittest discover -s tests`)
data/pilot_v0/     frozen P0 candidate data (see below)
data/contrib/      append-only extension sets, each with its own dataset.json
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

`data/annotation/calibration_A/` holds blank, neutral sheets. Structural hints are opt-in (`make-sheets --truth data/pilot_v0/candidate_truth.jsonl`); the answer-free worksheet alone never produces an "answer is empty" hint. Earlier sheets that told annotators all five segments were empty have been regenerated. `data/annotation/` is a local working area (it is git-ignored), so what each of its artefacts is — and, more importantly, is not — is documented in the committed `P0_标注链与金标冻结.md` rather than inside the folder.

### The human annotation chain

`make-sheets → parse-sheets → agreement → freeze-gold` is implemented and exercised end to end by `tests/test_annotation_pipeline.py`: the rendered sheets carry no answers, the parsed units satisfy the model-output contract, the agreement gate measures Cohen's κ only (AC1 is reported, never enforced), and the frozen gold is readable by the benchmark's own scorer.

Two properties are enforced rather than promised. The freeze **refuses generator or AI provenance** — the shipped `candidate_truth.jsonl` carries a generator truth container, marks every one of its 420 units `candidate_only: true`, and marks its dialogues `generated_unannotated_not_gold`, so that file can never back a `human_gold` claim even if the unit-level markers are stripped. And **one annotator cannot certify agreement**: freezing requires two distinct annotators, and the third reviewer who adjudicates the conflicts may not be either of them.

The checks now go beyond markers. The agreement report carries content hashes of both annotation sets; the freezer recomputes κ, the CI and the adjudication queue from those sets and cross-checks the report, then compares the labels themselves against every shipped generator truth file — both before and after adjudication. Deleting `candidate_only`, rewriting notes and changing `status` no longer changes the verdict. A second, distinct annotation set is mandatory, and two byte-identical sets are refused. Parsed sheets default to `provenance: "unknown"` until someone declares a source; the freezer will not promote that to human. `--distractors` optionally carries the rule-audited negative-side items into the gold (disclosed as `dataset_truth_not_human_annotated`); without it FMR is undefined and the manifest says so.

The freezer can verify that two distinct *annotator labels* exist; it cannot verify that two distinct *people* did the work. That limit is stated in `P0_标注链与金标冻结.md` §5.2 rather than papered over: the control for it is publishing the annotation records with identifiable annotators, and the frozen manifest records the labels so the claim can be audited.

What has *not* happened yet: no human has filled a sheet, so no κ has been measured and no gold has been frozen. The chain is a working instrument, not a result.

### The frozen pilot and the randomised extension

`data/pilot_v0/` is **frozen**: byte-identical regeneration is enforced in CI, and the set is never edited in place. It is also template-generated (generator v0), which is why a second set exists:

| | `data/pilot_v0/` (frozen) | `data/contrib/pilot-v1-randomized/` |
|---|---|---|
| Generator | deterministic template v0 | template v1, randomised layout + lexical variant pools |
| Dialogues | 40 (26–40 turns each) | 40 (120–130 turns each) |
| Turns / unique turn texts | 1 512 / 94 | 4 988 / 1 077 |
| Constraint units | 420 | 420 |
| Distinct slot→turn maps (35 non-empty dialogues) | 3 | 35 |
| Behaviour probes | 70, all `related: true` | 105, of which **35 are `related: false`** |
| Max probe lexical overlap | — | 0.1143 (ceiling 0.25) |
| Evidence grade | `generator_truth_rule_audited` | `generator_truth_rule_audited` |

Neither set is human gold, and neither has been through the eleven-step generation pipeline. The extension set exists for one concrete reason: the frozen pilot's 70 probes are all `related: true`, so dimension B's false-trigger denominator is zero and B has **no usable false-trigger estimate** from the pilot. The randomised set's 35 `related: false` control probes make that rate computable. Regeneration is byte-identical (`configs/generator.v1.json`, seed `20261120`).

The frozen pilot's template reuse is still severe and is the reason a second set exists: 35 of its 40 dialogues share only **3** distinct slot→turn layouts, and its 1 512 turns contain just **94** unique texts. The randomised set spreads the same slot structure across 35 distinct layouts and raises unique turn texts to 1 077. That is progress, not a solution — 1 077 unique texts over 4 988 turns is still template-generated speech, and the extension set is an append-only contribution, not a replacement for the frozen pilot.

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

### Executor probes (dimension B)

`run-probes` is the command-line entry point for the three-arm / four-background conditions. It consumes the induction predictions, so a probe whose target constraint was not induced is marked unscorable instead of silently becoming a "failure" data point.

```bash
PYTHONPATH=src python -m context_benchmark run-probes \
  --runtime configs/runtime.mine.json \
  --tasks data/pilot_v0/tasks.jsonl \
  --truth data/pilot_v0/candidate_truth.jsonl \
  --predictions runs/preds.jsonl \
  --executor-prompt prompts/executor_v2.txt \
  --output runs/probes.jsonl \
  --logs runs/probe_calls.jsonl \
  --scores runs/counterfactual.json
```

`false_trigger_rate` counts only probes explicitly marked `related: false`, and fires when the model follows the injected constraint in that unrelated situation (`true > none`). With no such probes it returns a null value with a zero denominator, not a misleading 0. The frozen pilot's 70 probes are all marked `related: true`, so B has **no usable false-trigger estimate** from that set. The randomised extension set (`data/contrib/pilot-v1-randomized`) carries 35 `related: false` control probes, so the rate becomes computable there; a contributed generator config can set `"add_unrelated_probes": true` (with its own `dataset_version` and seed) to do the same. The frozen pilot is unaffected.

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

The validator does not take the package's word for anything it can recompute from the bytes the package ships:

- **Denominators are recounted** from the packaged `predictions.jsonl`, not read from the manifest. The declared `scored_n`/`failed_n` must match the recount; when the dataset is reachable the semantic re-validation is authoritative and must match exactly.
- **Declared scores are recomputed** from the packaged predictions and the resolvable dataset. A package that claims scores must say how: `scores_verification: "reproduced_locally"` (the validator recomputed them and they match, within 1e-6) or `"not_reproducible_locally"` (the dataset is not reachable from the validator's machine — the package must then say so in `declared_limitations`, and the numbers may not be presented as verified). A mismatch is a hard failure.
- **Evidence hashes** of the canonical repository files (scorer, config, prompts) must match the installed tree, so changing the scorer invalidates every submission that claims reproducibility.

It also rejects unsupported versions, unpinned models, fake `human_gold`, missing compliance notes for real data, and score declarations that omit the primary dimension. It exits 1 on failure and runs in CI. Contract: `schemas/submission.schema.json`; implementation: `src/context_benchmark/manifest.py`.

A `human_gold` claim is checked against the **shipped bytes**, not just the annotation report's word: the packaged gold file is scanned for generator and AI provenance markers, so renaming `candidate_truth.jsonl` to `gold.jsonl` and writing a report that claims two human annotators and a plausible κ is rejected. The check is also **content-level**: the gold's unit-content fingerprint is compared with every generator truth file in the repository, so stripping the markers off that same file is rejected too (`fake_human_gold`). (The frozen gold legitimately carries a `candidate_truth` key because the scorer reads it — that key name is a file format, not a provenance claim, and the container rule is switched off for rows declaring `gold_provenance: "human"`.)

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
| `P0_标注链与金标冻结.md` | The annotation chain end to end: sheet format, agreement gates, provenance rules, and what each artefact is and is not |
| `P0_预检报告_AI对照.md` | What an AI cross-check found, including two ontology errors |

## Known limitations

1. Synthetic dialogues are template-generated; lexical diversity is low and the implicit-constraint cues may be more regular than real user behaviour. The randomised extension set raises unique turn texts from 94 to 1 077, but 1 077 unique texts over 4 988 turns is still template speech.
2. The text matcher, the anchor gate and the new semantic guard are **uncalibrated heuristics**. They block the known garbage-text and semantic-inversion exploits, but they are usable for engineering checks only — not for reporting model scores. There is no human-labelled calibration set, so their false-positive rate on *correct* answers is unknown.
3. `raw_turns` is a `first_recent_v0` baseline, not a reproduction of the selection policy in arXiv:2609.34227.
4. Only 40 dialogues per set. Anything inferred from them is a pilot signal.
5. No external reproduction, no pre-registration, no peer review.
6. The six-axis set is a **structural re-registration**, not six independently validated constructs. D was split from A and E/F renamed from R1/R2; their orthogonality has not been measured because the P1 correlation matrix has not been run. A pairwise r > 0.8 would merge them back.
7. Dimension B now has a CLI, but no real model results exist. The frozen pilot's probes are all `related: true`, so `false_trigger_rate` has a null denominator there; the randomised extension set supplies 35 `related: false` controls, but it has never been run against a model either.
8. Dimensions E/F now have runnable CLIs (`run-survival-trace` + `survival`, and `run-r2`), so they can produce traces and reports end-to-end. They are still **attribution-only** and never enter the ranking. The token counter for F is a `chars/4` heuristic until a model-specific tokenizer is wired in, and E's per-turn probes are a simple engineering construction, not a validated survival instrument. Treat both as P1 pilots, not reported results.
9. The oracle self-check reports `fama_macro = 0.875`, not 1.0, and that is deliberate: five empty-constraint dialogues have an undefined FAMA, and the macro mean uses a fixed denominator of all 40 dialogues so that a model producing nothing cannot improve the average by disappearing. The empty-output accuracy is reported separately at 5/5.
10. The generator is still a **template**, not the eleven-step pipeline with a semantic-disconnect filter. The lexical filter that is shipped is a character-bigram approximation and the config says so explicitly; it is not BM25+MPNet and must not be reported as one.
11. The human annotation chain is implemented and verified end to end, but **no human has filled a sheet**. No κ has been measured, no gold has been frozen, and — the consequence that matters for scoring — the text gate and the semantic guard still have **no human-labelled calibration set**, so their false-positive rate on correct answers remains unknown. The boundary tags (R1–R8) are collected from annotators rather than inferred, because on the shipped pilot 113 units share the shape `one_off`+`task` and only 35 of them are R1: any field-based derivation would mislabel 35 % of the units.

## License

MIT. See [LICENSE](LICENSE).
