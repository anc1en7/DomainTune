# DomainTune — LLM Fine-Tuning Project Specification

**Version:** 1.0  
**Status:** Implementation-ready proposal  
**Project type:** Applied ML / LLM fine-tuning / MLOps  
**Estimated duration:** 6–8 weeks (adjust to available compute)  
**Initial hardware:** Intel i5-12450H, RTX 3050 laptop GPU, 16 GB RAM; verify actual VRAM first  
**Primary stack:** Python, PyTorch, Hugging Face Transformers/Datasets/PEFT, TRL, scikit-learn, MLflow, FastAPI, Docker

> **Core goal:** Fine-tune a small instruction-tuned language model to classify banking customer-support messages into one of 77 verified intent classes; compare it fairly against baselines; measure training and inference trade-offs; deploy the best validated model with reproducible evaluation.

## 1. Executive summary

DomainTune is a complete machine-learning engineering project, **not just an API wrapper**. It demonstrates dataset validation, data leakage controls, supervised fine-tuning, LoRA/QLoRA, hyperparameter experimentation, quantitative evaluation, error analysis, and model deployment. The initial benchmark uses **BANKING77**, which has **13,083 labeled utterances across 77 intent classes**. The model candidate is **Qwen2.5-0.5B-Instruct**; use a larger model only after measuring local resource needs. Budget is zero; paid APIs, cloud GPUs, and billable hosting are excluded.

### 1.1 Questions the project must answer

1. Does fine-tuning improve **held-out macro-F1** over zero-/few-shot prompting with the unchanged base model?
2. How does the result compare with a **non-LLM baseline** (TF-IDF + linear classifier)?
3. What changes with **10%, 25%, 50%, and 100%** of the training data?
4. What are the trade-offs in **VRAM, wall-clock training time, inference latency, and prediction validity**?
5. Which intent categories remain difficult, and why?

### 1.2 Measurable definition of done

- [ ] A deterministic and documented dataset pipeline with **no test-set tuning**.
- [ ] At least three evaluated systems: TF-IDF baseline, unchanged instruction LLM, and LoRA-trained LLM.
- [ ] One completed training run with saved PEFT adapter, configuration, log, and exact dataset/model revisions.
- [ ] Test accuracy, macro-F1, per-class F1, confusion matrix, JSON/label validity, latency, throughput, and peak memory recorded.
- [ ] Ablation across at least two data fractions **or** two PEFT configurations, depending on compute.
- [ ] FastAPI `/predict` endpoint with strict schema and error responses; container build documented.
- [ ] Automated tests and a README enabling another engineer to replicate the final experiment.
- [ ] A results report describing **both successes and failed experiments**.

**Aspirational targets** (not promises): macro-F1 ≥ 0.85, accuracy ≥ 85%, output-label validity ≥ 99%, and ≥ 5 percentage-point macro-F1 gain over the unchanged model. If these are missed, publish the real measurements and analysis rather than manipulating the benchmark.

## 2. Product and research scope

### In scope

- Single-utterance intent classification into one of the BANKING77 labels.
- Supervised instruction/completion training with parameter-efficient adapters.
- Fair comparisons on a fixed, untouched test split.
- API-based inference and reproducible model packaging.
- Local GPU feasibility assessment; reduce workload if local resources are insufficient.

### Out of scope in v1

- Transactions, payment execution, real banking-system integrations, or customer-data access.
- Fraud scoring, urgency prediction, refund authorization, and policy recommendations: **BANKING77 does not provide ground-truth labels for these**.
- Training from scratch, large-scale distributed training, multilingual intent handling, RAG, or multi-agent systems.

**Data rule:** Use public, appropriately licensed data only. Avoid real customer PII. Check the dataset and model licenses before publication/distribution.

## 3. Recommended architecture

```text
BANKING77 (pinned revision)
   |
   v
Data audit -> normalization -> dedup/leakage checks -> train/val/test manifest
   |                                              |
   |                                              +--> frozen test set
   v
Instruction/label mapping + tokenizer
   |                         
   +--> TF-IDF + LogisticRegression baseline
   +--> Unchanged Qwen 0.5B inference baseline
   +--> PEFT LoRA/QLoRA training --> adapter checkpoints
                                      |
                                      v
                             Validation + run tracking
                                      |
                                      v
                           Champion model selection
                                      |
                                      v
                  Frozen test evaluation + error analysis
                                      |
                                      v
                         FastAPI inference service
                                      |
                                      v
                    Docker + health check + test suite
```

### Core modules

| Component | Purpose | Expected artifact |
|---|---|---|
| Data ingestion | Fetch pinned BANKING77 revision | `data/raw/` manifest |
| Data audit | Count classes, nulls, duplicates, overlaps | `reports/data_audit.md` |
| Prompt builder | Consistent training/inference format | `src/domaintune/prompts.py` |
| Baselines | Classical + unchanged LLM benchmarks | `reports/baselines.json` |
| PEFT training | Train adapters; log runs | `outputs/adapters/<run_id>/` |
| Evaluator | Macro-F1, class metrics, invalid outputs | `reports/evaluation/` |
| Experiment tracker | Parameters, metrics, hardware, artifacts | MLflow runs |
| Model serving | Validated JSON prediction endpoint | `app/main.py` |
| Tests | Contract, unit, integration, regression | `tests/` |

## 4. Dataset specification

### 4.1 Dataset

**BANKING77** is a dataset of banking-related customer-service utterances labeled with 77 intent classes. It has a published train/test split; verify and pin the exact dataset version and split sizes at ingestion time. It contains **intent labels only**, not severity, next action, or rationale ground truth.

Canonical internal record:

```json
{
  "sample_id": "sha256:...",
  "text": "I think I was charged twice for a card payment",
  "intent_id": 0,
  "intent_name": "card_payment_fee_charged",
  "source_split": "train",
  "dataset_revision": "<pinned-revision>"
}
```

`intent_name` above is an *illustrative schema value*; use the exact dataset label names and ID mapping as loaded, never an invented mapping.

### 4.2 Split policy

1. Preserve the official test split, which is **never used for prompt tuning, model selection, or hyperparameter search**.
2. Split the official training partition into **train and validation**, e.g. stratified 90/10 with a fixed seed.
3. Audit exact/normalized duplicate texts and near-duplicates across splits; report leakage risks and define a documented handling rule before training.
4. Fit any learned preprocessing and vectorizers only on training data.
5. Run final test evaluation **once per chosen final model** after locking configuration. If rerunning, label follow-up analysis as exploratory rather than new unbiased evidence.

### 4.3 Data validation checks

- [ ] All records have nonempty text.
- [ ] Every label ID maps to exactly one canonical class name.
- [ ] No train/validation/test overlap under normalized-text hash after documented dedup policy.
- [ ] Check class counts and text-length distribution.
- [ ] Inspect at least 10 examples per problematic class pair.
- [ ] Record dataset citation, revision, license, checksum, split seed, and preprocessing version.

### 4.4 Training instruction format

Use one fixed prompt template across unchanged-model inference and fine-tuning:

```text
System: You classify banking support messages. Respond with ONLY valid JSON.
Choose exactly one intent from the provided allowed labels.
Do not invent a label and do not include explanations.

User: Classify this support message:
"{text}"

Assistant: {"intent": "{canonical_label}"}
```

**Practical consideration:** Putting 77 labels in the prompt consumes tokens. Compare a compact label-ID schema versus full label names in *validation-only* experiments. Lock the best schema before final testing. Use the model's official chat template for actual tokenization.

### 4.5 Loss masking

Train on the assistant response, not on the user's prompt tokens. Confirm the training framework's assistant-only/completion-only loss behavior with a small test batch; do not assume masking is configured correctly by default.

## 5. Model and training design

### 5.1 Candidate model

- **Primary:** `Qwen/Qwen2.5-0.5B-Instruct` (pin exact model revision).
- **Optional comparison:** `Qwen/Qwen2.5-1.5B-Instruct`, only if memory/compute permits.
- Review the current model card for supported Transformers version, prompt/chat formatting, and distribution license.

### 5.2 Baselines

| Baseline | How to implement | Why |
|---|---|---|
| Majority class | Always predict most frequent training label | Sanity check |
| TF-IDF + logistic regression | Character/word n-grams; tune on validation | Strong inexpensive non-LLM reference |
| Unchanged instruction LLM | Zero-shot prompted classification with identical label contract | Measures value of fine-tuning |
| Optional few-shot LLM | Fixed training-derived examples, validation-selected prompt | Stronger prompt-only comparison |

All baselines must use the **same final held-out test examples** and identical metric calculation. For generation models, record invalid output rate and apply the same deterministic output parser.

### 5.3 PEFT training configuration (starting point, not guaranteed optimal)

| Setting | Proposed initial value |
|---|---|
| Adaptation | LoRA (start with BF16/FP16 depending on hardware); QLoRA if supported |
| LoRA rank `r` | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` initially; verify actual module names |
| Epochs | 2–3, with validation checkpoints |
| Learning rate | 2e-4 initial candidate; tune on validation |
| Micro-batch | 1 |
| Gradient accumulation | 8–16 |
| Sequence length | 256 initially; measure truncation, increase only if necessary |
| Optimizer | AdamW or supported 8-bit optimizer |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Seed | 42; repeat best config with extra seeds if compute permits |
| Precision | Hardware-supported FP16/BF16; confirm GPU capabilities |
| Gradient checkpointing | Enabled when memory-constrained |
| Evaluation cadence | At least once per epoch |
| Checkpoint selection | Best validation macro-F1, not test score |

**Important:** QLoRA relies on compatible 4-bit GPU kernels and libraries. Laptop GPU/CUDA combinations can be difficult; test a minimal 20-step run first. Fall back to standard LoRA or reduce the workload without truncating labels or targets. Document local hardware limitations if necessary. Do not report parameter-count or memory savings without measuring them.

### 5.4 Controlled experiments

Minimal matrix:

| ID | Model/setting | Dataset amount | Purpose |
|---|---|---|---|
| B0 | Majority class | 100% training | Sanity |
| B1 | TF-IDF + logistic regression | 100% training | Classical baseline |
| B2 | Unchanged Qwen 0.5B | None | Prompt baseline |
| F1 | LoRA rank 8 | 100% training | Primary fine-tune |
| F2 | LoRA rank 16 | 100% training | Rank ablation |
| D1 | Best LoRA setting | 10% training | Data efficiency |
| D2 | Best LoRA setting | 25% training | Data efficiency |
| D3 | Best LoRA setting | 50% training | Data efficiency |
| D4 | Best LoRA setting | 100% training | Data efficiency |

Sample stratified *within the training split* for D1–D4. If compute is limited, prioritize B0/B1/B2/F1, then one ablation. Avoid changing multiple variables in a single comparison.

## 6. Evaluation protocol

### 6.1 Primary metrics

- **Macro-F1 (primary selection metric):** Treats each of 77 classes equally.
- **Accuracy:** Percentage of correctly classified messages.
- **Per-class precision, recall, F1, and support.**
- **Confusion matrix:** Especially similar intent pairs.
- **Validity:** Parseable JSON percentage; canonical allowed-label percentage.

### 6.2 Operational metrics

- P50/P95 latency measured on the same hardware and batch size.
- Throughput (requests/sec) at batch size 1 and one documented batch scenario.
- Peak allocated/reserved GPU memory during training and inference.
- Adapter size, model load time, total training duration.
- Training tokens/sec if available; approximate compute/cost only with explicit assumptions.

### 6.3 Parsing and failure policy

For each prediction:

1. Attempt strict JSON parsing.
2. Require exactly the documented `intent` field with canonical label value.
3. Reject unknown or multiple labels; no silent fuzzy matching in evaluation.
4. Count rejected output as **invalid** and **incorrect** in headline test metrics.
5. In the production API, return an explicit error or abstention status rather than inventing a valid prediction.

If constrained decoding is implemented, evaluate that configuration separately; do not compare its validity numbers to unconstrained baselines without stating the change.

### 6.4 Error analysis

Produce:

- Top 15 confused class pairs with supporting anonymized/public examples.
- Per-class F1 sorted ascending.
- Failure categories: semantic ambiguity, short message, terminology confusion, formatting error, truncated input.
- Bootstrap confidence interval for accuracy/macro-F1 if feasible; document method and sample size.
- Comparison of qualitative examples for baseline versus fine-tuned model.

### 6.5 Reproducibility manifest

Every run must record:

```yaml
run_id: <run-id>
seed: 42
model_id: Qwen/Qwen2.5-0.5B-Instruct
model_revision: <commit-hash>
dataset_id: banking77
dataset_revision: <commit-hash>
train_split_manifest: <sha256>
prompt_template_version: v1
adapter_config: configs/lora_r8.yaml
framework_versions: <from-lockfile>
gpu: <measured-device>
cuda_version: <version>
metrics_path: reports/evaluation/<run-id>.json
```

## 7. API specification

### 7.1 `POST /predict`

**Request:**

```json
{
  "text": "Why was my card payment declined?"
}
```

**Successful response:**

```json
{
  "intent": "<canonical_label_from_dataset>",
  "model_version": "domaintune-v1",
  "latency_ms": 82.4
}
```

`latency_ms` is illustrative, not a measured target. Do not expose a model-generated confidence score as calibrated probability. Confidence can be added in v2 only after a documented calibration method.

**Errors:**

| Status | Meaning |
|---|---|
| `400` | Empty text or malformed input |
| `413` | Request too large |
| `422` | Schema validation failure |
| `503` | Model unavailable/not loaded |
| `500` | Unhandled inference failure (without sensitive internals) |

**Input contract:** string `text`, 1–2,000 characters initially; enforce token limit independently and return a clear error if exceeded. Never silently truncate user input without recording it.

### 7.2 Other endpoints

- `GET /health/live`: process health.
- `GET /health/ready`: model and tokenizer successfully loaded.
- `GET /model/info`: model ID, revision, adapter version, label count, allowed schema version (no secrets).
- `POST /predict/batch`: optional v1.1 after validating single-request endpoint.

### 7.3 Serving requirements

- Pydantic schemas for inputs and outputs.
- Exactly one in-memory model instance per worker (avoid inadvertent GPU duplication).
- `torch.inference_mode()` and deterministic decoding for classification (`do_sample=False`).
- Centralized structured logs without raw customer-like text by default.
- Timeout, max payload, and concurrency limits.
- License and model-source attribution in README.

## 8. Repository structure

```text
domaintune/
├── README.md
├── pyproject.toml
├── uv.lock                         # or requirements lockfile
├── .env.example
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── lora_r8.yaml
│   └── lora_r16.yaml
├── data/
│   ├── raw/                        # gitignored
│   ├── processed/                  # gitignored
│   └── manifests/
├── src/domaintune/
│   ├── data.py
│   ├── audit.py
│   ├── prompts.py
│   ├── baselines.py
│   ├── train.py
│   ├── predict.py
│   ├── parse.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── tracking.py
├── app/
│   ├── main.py
│   └── schemas.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_baselines.py
│   ├── train_peft.py
│   ├── evaluate_model.py
│   └── benchmark_inference.py
├── tests/
│   ├── test_data.py
│   ├── test_prompts.py
│   ├── test_parser.py
│   ├── test_metrics.py
│   └── test_api.py
├── reports/
│   ├── data_audit.md
│   ├── experiment_log.md
│   └── final_report.md
├── Dockerfile
└── docker-compose.yml              # optional
```

Do not commit private access tokens, entire raw datasets where license/repository size prohibits, model weight files, or oversized MLflow artifacts. Store reproducible retrieval commands and hashes instead.

---

# 9. Phase-wise execution plan

Each phase has a specific **purpose, implementation tasks, deliverables, validation checklist, and exit gate**. Do not advance just because the code executes; advance when the evidence exists.

## Phase 0 — Environment and feasibility (Days 1–3)

**Goal:** Verify the exact compute environment and prove that the model can load and run.

**Tasks**

1. Initialize Git repository, Python environment, lockfile, and formatter/linter.
2. Record OS, Python, PyTorch, CUDA, Nvidia driver, GPU model, available VRAM, and system RAM.
3. Install compatible `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `scikit-learn`, `mlflow`, `fastapi`, `uvicorn`, `pytest`, and optionally `bitsandbytes`.
4. Load Qwen2.5-0.5B-Instruct; generate one deterministic response using its chat template.
5. Run a **20-step PEFT smoke train** on tiny synthetic/public examples; monitor VRAM and step time.
6. Make a configuration decision: local LoRA, compatible local QLoRA, or document a local hardware limitation. Budget is zero; cloud-GPU training is excluded.
7. Freeze dependency versions and create `docs/hardware-feasibility.md`.

**Deliverables:** Git repository, working environment, hardware report, model load test, 20-step smoke-test log.

**Exit gate:** Model inference works; a small PEFT training run finishes without out-of-memory; configuration and limitations are documented.

**Common failure:** Assuming 4-bit support just because CUDA is installed. Verify actual bitsandbytes compatibility and GPU capability.

## Phase 1 — Dataset ingestion and leakage audit (Days 4–7)

**Goal:** Create trustworthy, repeatable data splits and an exact label mapping.

**Tasks**

1. Download BANKING77 from a reputable dataset host and pin its revision.
2. Preserve original labels; create a canonical `id -> name` mapping.
3. Generate deterministic sample IDs and dataset hashes.
4. Inspect nulls, class balance, duplicates, lengths, and Unicode issues.
5. Create stratified validation data from official training records.
6. Check exact and normalized text overlap across train/validation/test.
7. Investigate near-duplicates; document any policy for removing or retaining them.
8. Save split manifests and implement tests asserting isolation.

**Deliverables:** `data/manifests/`, `reports/data_audit.md`, dataset loading/validation tests.

**Exit gate:** Every sample is traceable to a frozen split; all 77 labels are valid; leakage checks pass or outstanding risks are recorded.

**Critical:** Never select prompts or hyperparameters using the official test labels.

## Phase 2 — Baselines and evaluation harness (Week 2)

**Goal:** Establish what performance is achievable *without fine-tuning*.

**Tasks**

1. Implement majority-class sanity baseline.
2. Train TF-IDF + logistic regression on training partition; tune basic parameters on validation.
3. Define versioned prompt and strict output parser for the unchanged instruction-tuned LLM.
4. Evaluate zero-shot on validation, including invalid output percentage.
5. Optionally test few-shot prompt design using examples only from training data.
6. Implement macro-F1, accuracy, per-class metrics, confusion matrix, and error reports.
7. Confirm metric code against hand-computed cases and scikit-learn outputs.
8. Log baselines and settings to MLflow or comparable structured run storage.

**Deliverables:** Baseline scripts, metric tests, initial evaluation report, immutable prompt template.

**Exit gate:** B0, B1, and B2 results can be regenerated with one documented command each; invalid predictions are counted consistently.

**Important:** A well-tuned conventional classifier may outperform the small LLM. Report this plainly; it strengthens the comparison.

## Phase 3 — Training pipeline and primary LoRA experiment (Week 3)

**Goal:** Produce the first reproducible fine-tuned checkpoint.

**Tasks**

1. Convert examples to an instruction-response format with exact canonical labels.
2. Use official chat template and verify label tokenization.
3. Verify loss masking: only target assistant completion contributes to loss.
4. Configure LoRA with explicit target modules and a measured memory budget.
5. Train using fixed seed, gradient accumulation, validation checks, and checkpoint save.
6. Monitor train loss, validation macro-F1, VRAM, wall-clock duration, and step throughput.
7. Choose checkpoint based only on validation performance.
8. Store adapter, tokenizer/config references, run manifest, and full logs.

**Deliverables:** `outputs/adapters/<run_id>/`, training config, experiment logs, validation charts.

**Exit gate:** Fine-tuned adapter loads independently and improves on—or is fairly comparable to—the unchanged model on validation; evaluation script works end-to-end.

**Failure-handling policy:** If a run fails, capture error, GPU state, attempted configuration, and next change. Do not hide failed experiments.

## Phase 4 — Ablations and model optimization (Week 4)

**Goal:** Demonstrate that results reflect controlled decisions rather than a single lucky run.

**Tasks**

1. Compare LoRA rank 8 versus rank 16, keeping other factors fixed.
2. Optionally compare LoRA and QLoRA if hardware supports both.
3. Run stratified training-data fractions (10%, 25%, 50%, 100%), resource permitting.
4. Repeat best configuration with 2–3 seeds if compute is available.
5. Measure peak GPU memory, trainable parameters, elapsed time, and validation macro-F1.
6. Examine overfitting trends and label confusion.
7. Select champion configuration **only on validation evidence**.

**Deliverables:** Experiment comparison table, data-efficiency curve, memory/training-cost comparison, selection rationale.

**Exit gate:** At least one controlled ablation completed and champion model frozen. Explain any skipped experiment due to compute constraints.

## Phase 5 — Final held-out evaluation and error analysis (Week 5)

**Goal:** Produce credible evidence of generalization.

**Tasks**

1. Freeze checkpoint, tokenizer, parser, label mapping, and generation settings.
2. Run final evaluation against the official held-out test set.
3. Evaluate the matched baselines on the same examples.
4. Report macro-F1, accuracy, micro/weighted F1 (secondary), per-class metrics, and invalid rate.
5. Generate confusion matrix and worst-performing intent classes.
6. Inspect at least 25–50 wrong predictions; group errors by cause.
7. Compute uncertainty estimates when appropriate; note dataset and sampling limitations.
8. Measure end-to-end inference latency on specified hardware with warm-up and repeated runs.
9. Document whether model fine-tuning meaningfully beats the classical baseline.

**Deliverables:** `reports/final_evaluation.json`, confusion matrix, benchmark chart, `reports/error_analysis.md`.

**Exit gate:** Honest, reproducible test results published; conclusion explicitly answers RQ1–RQ5 (or identifies unanswered questions).

## Phase 6 — FastAPI inference and containerization (Week 6)

**Goal:** Serve the frozen model reliably as a typed inference service.

**Tasks**

1. Build model loader with one cached model instance.
2. Implement `/predict`, `/health/live`, `/health/ready`, and `/model/info`.
3. Validate request length and schema; return deterministic errors.
4. Use deterministic generation and strict label parsing.
5. Load model adapter and the exact base model revision used in evaluation.
6. Record latency and health metrics; do not log message text by default.
7. Implement unit tests for successful predictions, malformed input, unknown label, and missing model.
8. Build Docker image; document GPU runtime dependencies and optional CPU behavior.
9. Test clean-start deployment on a new environment where possible.

**Deliverables:** Runnable API, Dockerfile, `.env.example`, OpenAPI docs, API tests, deployment instructions.

**Exit gate:** The API returns a canonical label for valid test requests, rejects invalid inputs, and can be started from the README.

## Phase 7 — Reliability and reproducibility (Week 7)

**Goal:** Turn a working demo into an engineering-quality repository.

**Tasks**

1. Add CI for lint, type checks where practical, and fast unit/integration tests.
2. Keep large model downloads and GPU training out of basic CI; use mocks or small fixtures.
3. Add a regression fixture with stable known examples and parser edge cases.
4. Verify pinned package versions and artifact hashes.
5. Define model lifecycle/versioning and rollback procedure.
6. Add inference load test with concurrency and timeout constraints.
7. Write known limitations, license attribution, privacy policy, and hardware instructions.
8. Run clean-environment reproduction of the primary pipeline.

**Deliverables:** Green CI checks, regression report, deployment/reproduction guide, model card.

**Exit gate:** Another developer can understand the experiment and reproduce a representative result without undocumented manual steps.

## Phase 8 — Portfolio packaging and interview preparation (Week 8)

**Goal:** Convert the work into verifiable hiring evidence.

**Tasks**

1. Create an architecture diagram and 60–90 second demo video.
2. Publish a concise README with problem, methodology, baseline comparison, metrics, limitations, and deployment steps.
3. Add a model card: intended use, data, metrics, risk, limitations, hardware, licensing.
4. Produce a results table showing baseline vs LoRA vs ablation with actual numbers.
5. Document one major failure and its root cause.
6. Draft two concise resume bullets using measured outcomes.
7. Prepare verbal explanations of LoRA, attention projections, gradient accumulation, data leakage, macro-F1, and evaluation methodology.

**Deliverables:** Portfolio-ready GitHub repository, demo, model card, final report, resume bullets.

**Exit gate:** Every numeric claim in the README and resume is traceable to a saved run and evaluation artifact.

---

# 10. Verification and acceptance test plan

| ID | Test | Expected outcome |
|---|---|---|
| DATA-01 | Load pinned dataset | Same row count and label mapping on repeated run |
| DATA-02 | Split isolation | No prohibited duplicate overlap across partitions |
| DATA-03 | Label validation | Every label belongs to canonical 77-class set |
| PROMPT-01 | Prompt rendering | Exactly one correctly formatted user input |
| TRAIN-01 | Loss masking | Prompt tokens do not contribute to objective |
| TRAIN-02 | Adapter reload | Predictions possible after clean process reload |
| EVAL-01 | Metric unit tests | Match hand-computed values |
| EVAL-02 | Invalid output | Counted as invalid **and** incorrect |
| EVAL-03 | Test isolation | Test results never influence checkpoint selection |
| API-01 | Valid request | Valid schema with canonical intent |
| API-02 | Empty request | Client validation error |
| API-03 | Unknown generated label | Explicit error/abstention, never fabricated mapping |
| API-04 | Model unready | 503 response |
| PERF-01 | Benchmark script | Outputs hardware, warm-up, P50/P95, batch size |
| REPRO-01 | Environment recreate | Runs with pinned software and model revisions |

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Laptop VRAM insufficient | Training blocked | Smoke test; reduce workload; compatible local LoRA/QLoRA; document hardware limits |
| QLoRA dependency mismatch | Broken CUDA kernels | Pin compatible versions; fall back to LoRA |
| Model outputs invalid labels | Misleading accuracy | Strict parser, invalid-rate reporting, optional constrained decoding as separate configuration |
| Data leakage | Inflated metrics | Split manifests, duplicate audit, test quarantine |
| LLM worse than TF-IDF baseline | Weak primary hypothesis | Report honestly; perform class-error analysis and resource trade-off study |
| Training instability | Non-reproducible results | Fixed config/seed, logged runs, checkpoints, additional seeds |
| Public dataset not representative of live support | Limited real-world validity | Explicit limitation; no real-world performance claims |
| Overengineered project scope | Missed completion | Freeze v1 requirements, defer UI/agents/RAG |
| Model/API security | Untrusted input, oversized payloads | Validation, size limits, no default raw-input logging |
| Licensing/redistribution constraints | Cannot share artifacts | Check licenses, cite creators, share scripts/adapters only when permitted |

## 12. Suggested development commands (interface targets)

Implement command entry points so these workflows are documented and stable:

```bash
# Data
python scripts/prepare_data.py --config configs/data.yaml

# Baselines
python scripts/run_baselines.py --config configs/baseline.yaml

# Train
python scripts/train_peft.py --config configs/lora_r8.yaml

# Evaluate selected model
python scripts/evaluate_model.py --checkpoint outputs/adapters/<run_id> --split test

# Benchmark inference
python scripts/benchmark_inference.py --checkpoint outputs/adapters/<run_id>

# Run API
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Test
pytest -q
```

These are **planned CLI contracts**, not commands for an existing finished implementation. Build scripts to match these signatures or update the README consistently.

## 13. Final reporting template

The final report should include:

1. **Problem and dataset**: task definition, class count, revisions, split rules.
2. **Baselines**: majority, TF-IDF, unchanged LLM and settings.
3. **Training**: PEFT configuration, optimizer, hardware, steps, duration, seed.
4. **Results table**: accuracy, macro-F1, invalid rate, latency, VRAM.
5. **Ablation**: at least one controlled comparison.
6. **Error analysis**: worst classes and confusing label pairs.
7. **Deployment**: endpoint contract, model load time, environment.
8. **Limitations**: dataset validity, domain transfer, prompt sensitivity, reproducibility caveats.
9. **Conclusion**: whether fine-tuning was worthwhile *relative to both baselines*.

Example results table (fill only with measured numbers):

| System | Test accuracy | Macro-F1 | Invalid % | P95 ms | Peak VRAM GB |
|---|---:|---:|---:|---:|---:|
| Majority class | [measured] | [measured] | N/A | [measured] | N/A |
| TF-IDF + logistic regression | [measured] | [measured] | 0 | [measured] | [measured] |
| Unchanged Qwen 0.5B | [measured] | [measured] | [measured] | [measured] | [measured] |
| LoRA rank 8 | [measured] | [measured] | [measured] | [measured] | [measured] |
| LoRA rank 16 | [measured] | [measured] | [measured] | [measured] | [measured] |

## 14. Resume-ready deliverables

Write bullets *after* experiments are finished. Examples with placeholders:

- **Fine-tuning:** “Fine-tuned a [parameter-count] instruction LLM with PyTorch and LoRA on [N] banking-intent samples; improved held-out macro-F1 from [baseline] to [result] across 77 classes.”
- **Evaluation:** “Built a reproducible ML benchmark comparing TF-IDF, zero-shot LLMs, and PEFT adapters; tracked [N] experiments with per-class error analysis, inference latency, and GPU memory measurements.”
- **Deployment:** “Deployed an adapter-based intent classifier through FastAPI and Docker, with schema-validated predictions, regression tests, and P95 latency of [measured] ms on [hardware].”

**Never claim training from scratch, advanced quantization improvements, calibrated confidence, or production traffic unless actually performed and verified.**

## 15. Reference documentation

Verify current dependency compatibility before installing:

- Qwen model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- BANKING77 dataset: https://huggingface.co/datasets/PolyAI/banking77
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- Hugging Face TRL SFT Trainer: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face Datasets: https://huggingface.co/docs/datasets/index
- PyTorch: https://pytorch.org/docs/stable/index.html
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- MLflow: https://mlflow.org/docs/latest/ml/
- FastAPI: https://fastapi.tiangolo.com/

---

**Recommended execution order:** Environment feasibility → dataset integrity → fair baselines → one real LoRA training run → controlled ablation → frozen test evaluation → deployment → portfolio. Do not prioritize the UI, an agent system, or a flashy demo ahead of the evaluation evidence.
