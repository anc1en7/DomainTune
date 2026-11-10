# Session handoff - 2026-11-10

## Status and stopping point

Phase 1 is complete. STOP HERE: the user requires stopping after each phase and
waiting before starting the next. Phase 2 has not started. No full training,
validation benchmarks, or official test evaluation has run. The API remains a
skeleton returning 503 when a model backend is needed.

Read AGENTS.md and DomainTune_LLM_Fine_Tuning_Project_Spec.md before future work.
The user also requires committing and pushing every crucial verified change.

## Constraints and workspace

- Zero budget: local compute and free public downloads only; no paid APIs,
  cloud GPUs, or billable hosting.
- Workspace: D:\Code\DomainTune. All project environments, downloads, caches,
  temporary files, data, weights, adapters, logs, and reports stay on D:.
- Dot-source scripts/environment.ps1 in every new PowerShell session before tools.
- Existing shared C: installations/caches are not project-owned; do not move/delete.
- Repository: https://github.com/anc1en7/DomainTune (private), main tracks origin/main.
- GitHub CLI is authenticated as anc1en7. Never print credentials.
- Sandbox blocks network and .git writes; use the normal escalation mechanism.

## Verified environment

Intel i5-12450H, RTX 3050 Laptop GPU with 4096 MiB VRAM, approximately 15.7 GiB RAM.
NVIDIA driver 610.62. Project-managed Python 3.12.13 and .venv are on D:.
ML extra installed: torch 2.11.0+cu128, transformers 4.57.6, datasets 4.8.5,
peft 0.21.2, accelerate 1.15.0. Exact packages are in uv.lock.
Windows PyTorch uses an explicit official CUDA 12.8 index. CUDA and BF16 verified.
D: had about 142 GiB free after dependency installation, before model/data retrieval.

## Phase 0 evidence

- scripts/phase0_smoke.py: executable local inference/training/reload probe.
- configs/phase0.yaml: immutable input revisions and upstream licenses.
- reports/phase0-smoke.json: successful fixed-shape run with exact settings and losses.
- reports/phase0-initial-smoke.json: initial successful run with excessive reservation.
- reports/phase0-labels.json: exact canonical 77-label array; array index is intent ID.
- docs/hardware-feasibility.md: measured results, limitations, reproduction commands.

Model: Qwen/Qwen2.5-0.5B-Instruct at
7ae557604adf67be50417f59c2c2f167def9a775 (upstream metadata: Apache-2.0).
Dataset: PolyAI/banking77 at
1846984189f2bc0ac51c6fc3a600fa9e1714e0c6 (dataset card: CC-BY-4.0).
Preserve attribution and review redistribution obligations before distributing artifacts.
No software code license has been selected.

Full-label chat plus completion lengths are 530-625 tokens across 10,003 official
training rows. All exceed 256 tokens. Proposed LoRA config now uses 640, but full-data
640-token memory feasibility remains to be checked later. Never truncate labels/targets.

Successful smoke configuration: BF16 standard LoRA r=8, alpha=16, dropout=0.05,
q_proj/v_proj, micro-batch 1, accumulation 16, AdamW lr=2e-4, seed 42,
gradient checkpointing, response-only loss, fixed padding to 551 tokens.
20 optimizer updates completed in 226.17 seconds; peak allocated 1.92 GiB,
peak reserved 2.15 GiB. Saved adapter reloaded with a fresh base model in the same
process and generated successfully. Adapter weights: 2,175,168 bytes.
All 32 examples are card_arrival: this is a mechanics probe, not quality evidence.
Initial unchanged-model output used unknown label status and was correctly rejected.
No baseline accuracy or macro-F1 has been measured.

Large artifacts stay ignored under outputs/phase0-fixed-shape/adapter and .local.
The initial run remains at outputs/phase0-attempt3. Early failed reports:
outputs/phase0 (sandbox network blocked), outputs/phase0-network (metadata access bug,
fixed). Initial dynamic-shape run reserved 4.93 GiB despite only 1.92 GiB live tensors;
fixed padding and clearing inference cache reduced reservation to 2.15 GiB.
Both successful runs are retained as evidence. The dataset builder cached both official
splits, but only training records were used; no test predictions or tuning occurred.

## Verification and cautions

All 18 existing tests passed with the ML stack installed; Ruff lint/format passed.
Starlette emits an upstream TestClient deprecation warning. Pytest cache writes can
be sandbox-blocked; use -p no:cacheprovider for local verification.
Docker remains untested because its Linux daemon was stopped. Before Docker builds,
configure Docker Desktop disk image storage on D: separately.
No tokenizer/model loading requires paid APIs or remote inference.
Configs outside the Phase 0 runner remain proposals; the spec's future CLIs are not
implemented. Model serving, full evaluation, ingestion/splits, and training CLIs remain.

## Phase 1 evidence

Implemented `scripts/prepare_data.py`, `src/domaintune/data.py`, and audit module.
Run with configs/data.yaml; cached pinned data works with HF_HUB_OFFLINE=1 and
HF_DATASETS_OFFLINE=1. Offline fallback cache paths are checked for exact revision.
Repeated full runs produce identical processed split hashes. Manifests and source
checksums are in data/manifests; original texts remain ignored in data/processed.
Data audit: reports/data_audit.md. Prepared train/validation/test: 8992/1000/3080.
Excluded: 7 training rows matching normalized test text, 4 training duplicates.
Zero normalized overlap between prepared partitions; 137 cross-split near-duplicate
candidates retained as documented risk. Official test rows and labels untouched.
Reviewed candidates and ten training examples in each of six ambiguous classes.
30 tests pass; Ruff lint/format passes. No baseline or training run started.

## Resume after user authorizes Phase 2

1. Load the environment; verify clean Git state and existing tests.
2. Read the frozen data policy and verify manifest hashes before baselines.
3. Implement majority and TF-IDF baselines using only prepared training records.
4. Implement unchanged-model validation baseline and evaluation harness.
5. Select settings only using validation; never reuse the audit vectorizer.
6. Record reproducible baseline results and commit/push verified milestones.
7. Stop after Phase 2. Do not begin full LoRA training until requested.

Never tune prompts, hyperparameters, or checkpoints on official test results.
Count invalid outputs as incorrect. Never fabricate metrics or completed work.
