# DomainTune

Reproducible BANKING77 intent classification comparing TF-IDF, unchanged
Qwen2.5-0.5B-Instruct, and LoRA fine-tuning across 77 canonical intent classes.

**Status: Phase 3 continuation stopped for laptop temperature; epoch-1 adapter preserved.
Validation accuracy / macro-F1: TF-IDF 89.2% / 0.891;
unchanged Qwen 13.7% / 0.152, with 60.8% invalid outputs. No official test
evaluation has run.**
See the [baseline report](reports/baselines.md) for settings and reproduction.
Frozen splits contain 8,992 training, 1,000 validation, and 3,080 official test rows.
Eleven training duplicates were excluded; 137 near-duplicate candidate pairs
remain documented risks.
See the [data audit](reports/data_audit.md) for reproduction and leakage policy.
See [hardware evidence](docs/hardware-feasibility.md). The [project specification](DomainTune_LLM_Fine_Tuning_Project_Spec.md)
defines the full research plan and acceptance criteria.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```powershell
. .\scripts\environment.ps1
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Dot-source `scripts/environment.ps1` in every new PowerShell session before running
tools. It keeps uv/pip, Hugging Face, PyTorch, CUDA, and temporary caches under
`D:\Code\DomainTune\.local`, the environment under `.venv`, and MLflow runs under
`mlruns`. These settings apply to the current process and child processes only.
Use `uv python install 3.12` after loading the script to install a project-local
Python runtime on D:; use `uv sync --locked --python 3.12 --managed-python` to use it.

**Budget: zero.** All training and inference run locally; no paid APIs or cloud GPUs.
Free public model/dataset downloads still need internet access. If memory is tight,
reduce the workload and document limits rather than moving to paid compute.
Docker Desktop stores images separately: configure its disk image location on D:
before building containers. This script does not move Docker or system tools.

Open `/docs` for the API contract. Liveness returns 200. Readiness, model info, and
valid prediction requests return 503 until a validated model backend is implemented.
Invalid request schemas return 422. This skeleton does not yet enforce the full
serving contract (token limits, payload limits, timeouts, concurrency, or inference).

Optional ML dependencies are resolved in `uv.lock`, but are not installed by default:

```powershell
uv sync --locked --extra ml
```

Validate CUDA/PyTorch compatibility before installing the ML extra for GPU work.
No quantization backend is configured. See [hardware notes](docs/hardware-feasibility.md).

## Included

- Installable Python package and locked environment, Ruff, pytest, GitHub Actions.
- Strict JSON intent parser, versioned chat messages, invalid-aware classification metrics.
- FastAPI skeleton, proposed experiment configs, CPU service container definition.
- Original specification and report templates; generated data and weights are ignored.

```powershell
docker build -t domaintune .
docker run --rm -p 8000:8000 domaintune
```

## Next milestones

1. Phase 1 complete: ingestion, canonical labels, deterministic splits, and leakage audit.
2. Phase 2 complete: majority, TF-IDF, and unchanged-model validation baselines.
3. Train LoRA with verified completion-only loss and complete a controlled ablation.
4. Freeze the champion, evaluate the untouched official test split, then implement serving.

`scripts/prepare_data.py --config configs/data.yaml` and
`scripts/run_baselines.py --config configs/baseline.yaml` are implemented (require
the ML extra). See the baseline report for separate classical and LLM commands.
`scripts/train_peft.py --config configs/lora_r8.yaml` implements primary LoRA
training; `scripts/evaluate_model.py --checkpoint <path> --split validation`
verifies adapter reload in a fresh process. See [training protocol](reports/training.md).
Final test evaluation and benchmark CLIs remain future work.
Select prompts and checkpoints on validation only. Count invalid predictions as incorrect.
Never silently truncate messages or invent canonical labels.

## Sources and licensing

- Dataset: [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77).
- Candidate model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

Pinned revisions are in `configs/phase0.yaml`. Upstream model metadata declares
Apache-2.0; the dataset card declares CC-BY-4.0. Preserve upstream attribution and
review redistribution requirements before publishing model or dataset artifacts.
No dataset or model artifacts are distributed here. No code license has been selected.
