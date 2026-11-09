# Session handoff — 2026-11-09

## Goal and current state

DomainTune compares classical and small instruction-LLM baselines against LoRA
fine-tuning for BANKING77's 77 banking intent classes. The full plan is in
`DomainTune_LLM_Fine_Tuning_Project_Spec.md`; read it and `AGENTS.md` before work.

The user requested project initialization locally and on GitHub. That is complete.
This is a scaffold, not a completed ML pipeline. No dataset or model has been
downloaded, no training has run, and no benchmark results exist.

## Nonnegotiable user constraints

- Zero budget. No paid APIs, cloud GPUs, or billable services. Run locally and use
  free public downloads. Reduce workload if hardware is insufficient.
- Keep project runtimes, environments, caches, temporary files, datasets, weights,
  checkpoints, logs, and reports on D:. C: has limited space.
- Load `scripts/environment.ps1` in every new PowerShell session before tools.
  Its environment variables are process-scoped, not global or automatically persistent.
- Do not delete/move shared system installations or caches without checking scope.
- GitHub repository is private; retain that setting unless the user requests otherwise.

## Workspace and repository

- Workspace: `D:\Code\DomainTune` (Windows, PowerShell).
- Repository: https://github.com/anc1en7/DomainTune
- Branch: `main`, tracking `origin/main`.
- GitHub CLI is authenticated as `anc1en7`; never print credentials.
- Initial commit: `edf40e6` — project scaffold.
- Storage/budget commit: `13dbb4b` — local-only constraints and D: environment.
- Worktree was clean before writing this handoff.

## What exists

- `pyproject.toml`, `uv.lock`, editable package `src/domaintune`, Python environment.
- Strict JSON parser rejecting extra fields, duplicate keys, and noncanonical labels.
- Versioned chat prompt builder; metrics counting invalid predictions as incorrect.
- FastAPI skeleton with request schemas and health/model/prediction routes.
- Proposed data, baseline, LoRA rank-8 and rank-16 configs (not executable pipelines).
- Tests, Ruff, GitHub Actions, Dockerfile, ignored data/artifact directories.
- README, hardware notes, and clearly pending report templates.
- `AGENTS.md` stores user constraints for future agents.

The API deliberately returns 503 for readiness, model info, and valid predictions:
there is no model backend. Liveness returns 200; invalid request schemas return 422.
Token/body limits, concurrency, timeouts, inference, full evaluation, ingestion,
training, and benchmark CLIs still need implementation. Specification section 12
contains future CLI contracts, not currently available scripts.

## Hardware and installed environment

- CPU: Intel Core i5-12450H.
- GPU: RTX 3050 Laptop GPU, 4096 MiB VRAM; NVIDIA driver 610.62.
- RAM: 16,869,548,032 bytes (about 15.7 GiB).
- Project-managed Python: 3.12.13 under `.local\python` on D:.
- `.venv` was rebuilt to use that D: Python (previously used a C: installation).
- uv 0.11.8; Git 2.47.0.windows.1; Docker CLI is installed.
- Base/dev dependencies installed. Optional `ml` extra is locked but NOT installed.
- PyTorch CUDA build, BF16 support, model loading, and training feasibility unverified.

## Resume commands

```powershell
Set-Location D:\Code\DomainTune
. .\scripts\environment.ps1
uv sync --locked --python 3.12 --managed-python
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The environment script points uv/pip, Hugging Face hub/datasets/assets/Xet, PyTorch,
extensions/compiler, CUDA, matplotlib, and temporary caches into `.local` on D:.
MLflow tracking points at `D:\Code\DomainTune\mlruns`; `.venv` stays in the workspace.
These paths are ignored by Git. Run all future installation/training commands with
the script loaded. Do not install the large ML extra before checking GPU packaging.

## Verification and caveats

- Initial environment: 18 tests passed; Ruff lint and formatting passed.
- After D: environment rebuild: API import, Python base path, temporary directory,
  and Ruff checks verified. Full tests were not rerun locally after that rebuild.
- GitHub CI passed for both scaffold and storage commits, including tests/format/lint.
  Latest verified run: https://github.com/anc1en7/DomainTune/actions/runs/42661195425
- One upstream Starlette warning about httpx TestClient deprecation was observed;
  tests still passed.
- Docker build not tested: Docker Desktop Linux daemon was stopped. Before using
  Docker, separately configure Docker Desktop disk image storage on D:; the project
  environment script cannot redirect it.
- The FIRST setup used shared uv cache `C:\Users\harsh\AppData\Local\uv\cache`.
  It was left intact because other projects may use it. Future project downloads
  use D:. Existing system tools can still reside on C:.
- No software code license selected; dataset license verification pending. The Qwen
  model card was checked and advertises Apache-2.0. Pin and record exact revisions
  and upstream licenses before experiments/distribution.

## Recommended next action: finish Phase 0

1. Load the D: environment script; inspect current hardware and free storage.
2. Select/install a CUDA-enabled PyTorch build and compatible ML dependencies.
   Check the existing lockfile and package index choice rather than assuming
   `uv sync --extra ml` installs the right CUDA build.
3. Pin Qwen and dataset revisions; record provenance and licenses.
4. Load Qwen with its official chat template and run deterministic inference.
5. Measure prompt token lengths, then run a 20-step LoRA smoke test with verified
   response-only loss. Record peak VRAM, duration, and failures.
6. Update hardware feasibility notes with evidence. No cloud fallback.

Suggested starting settings, not a measured optimum:

| Setting | Candidate |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Adaptation | Standard LoRA; try 4-bit QLoRA only if needed and compatible |
| Precision | FP16 initially; verify BF16 support |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | q_proj, v_proj; verify names |
| Micro-batch / accumulation | 1 / 16 |
| Learning rate | 2e-4 |
| Epochs | Start with 2, choose using validation macro-F1 |
| Memory controls | Gradient checkpointing; disable cache during training |
| Optimizer | Standard AdamW |
| First data experiment | Stratified 10% of training data after smoke test |

Important: 256 tokens is only a proposal. All 77 full label names may exceed the
budget. Measure the complete chat prompt plus response; validate a compact label-ID
schema if needed, keep it identical across unchanged/fine-tuned comparisons, and
never truncate away labels or the assistant target. Save the exact canonical mapping.

## Remaining phases

0. Environment: inference and measured smoke train (initialization done).
1. Data: pinned ingestion, canonical labels, train/validation split, leakage audit.
2. Baselines: majority, TF-IDF, unchanged Qwen with shared evaluation contract.
3. Training: completion-only LoRA with validation checkpoint selection.
4. Ablation: rank or data-fraction comparison within local compute limits.
5. Final evaluation: freeze champion, untouched official test, errors and performance.
6. Serving: real model backend, strict API errors/limits, Docker verification.
7. Reliability: reproducibility, regression checks, deployment documentation.
8. Portfolio: actual results, model card, demo, evidence-backed resume bullets.

Never tune prompts, hyperparameters, or checkpoints on the official test set. Count
invalid outputs as incorrect. Never fabricate labels, metrics, or completed work.
The user ended the session; this handoff is for resumption, not authorization to
start a long training run before they return.
