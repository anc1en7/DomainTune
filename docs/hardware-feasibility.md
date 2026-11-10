# Initial hardware inventory

Observed on 2026-11-09:

- OS: Windows (PowerShell workspace).
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU.
- VRAM: 4096 MiB, as reported by `nvidia-smi`.
- CPU: 12th Gen Intel Core i5-12450H.
- System RAM: 16,869,548,032 bytes (approximately 15.7 GiB).
- NVIDIA driver: 610.62.
- Installed Python: 3.12.4 and 3.13.9; project environment uses 3.12.
- uv: 0.11.8; Git: 2.47.0.windows.1.

Verified on 2026-11-10 after installing the locked ML extra:

- PyTorch 2.11.0+cu128 (explicit official CUDA 12.8 wheel source on Windows).
- `torch.cuda.is_available()` returned true and detected the RTX 3050 Laptop GPU.
- `torch.cuda.is_bf16_supported()` returned true; actual BF16 training is pending.
- D: free storage after installation: 152,069,365,760 bytes (about 142 GiB).
- All 18 existing tests and Ruff checks passed with ML dependencies installed.

Initial smoke run on 2026-11-10 (`reports/phase0-initial-smoke.json`):

- Full-label chat plus completion: 530-625 tokens across all 10,003 official training rows;
  every row exceeds 256 tokens. No truncation was used.
- BF16 rank-8 LoRA completed 20 optimizer steps (16 micro-batches each) in 226.83 seconds.
- Prompt/completion token-prefix and response-only loss masking checks passed.
- Trainable parameters: 540,672 of 494,573,440 total; adapter weights: 2,175,168 bytes.
- Adapter saved and reloaded with a fresh base model in the same process; generation worked.
- Peak allocated: 2,056,887,808 bytes (1.92 GiB); peak reserved: 5,297,405,952 bytes
  (4.93 GiB). Reserved memory exceeds physical VRAM, so this is not a clean 4 GiB fit.
- Initial unchanged-model output used an unknown label (`status`); the strict parser rejected
  it. This one-example probe is not a baseline accuracy measurement.
- Smoke examples were the first 32 training rows; low loss is not evidence of generalization.

Fixed-shape rerun passed (`reports/phase0-smoke.json`):

- Same pinned inputs, 32 examples, BF16 LoRA, and 20 optimizer updates.
- Pad to 551 tokens with an explicit attention mask and ignored padding labels;
  clear unused inference cache before training. No prompt or response truncation.
- Training: 226.17 seconds; warm cached model load: 1.61 seconds.
- Peak allocated: 2,056,892,416 bytes (1.92 GiB).
- Peak reserved: 2,306,867,200 bytes (2.15 GiB), below physical 4 GiB VRAM.
- Adapter reload and generation passed. Adapter weights are approximately 2.07 MiB.
- All 32 smoke examples have label `card_arrival`. This verifies the mechanics only;
  loss reduction and the reloaded prediction do not measure classification quality.

**Phase 0 exit gate passed for the measured 551-token configuration.** Use standard
BF16 LoRA locally; quantization is unnecessary for this smoke configuration.
The proposed 640-token full-data limit covers measured training rows but has not
yet been memory-tested; recheck it with representative labels before full training.
The first run's load time includes weight retrieval and is not a cached-load benchmark.
Official test examples were never used for generation, tuning, or training; the
Datasets builder did materialize both official splits in its local cache.
Driver-reported CUDA capability is not proof of a compatible installed PyTorch runtime.
The 4 GiB GPU requires an actual measured LoRA smoke test before choosing settings.
User constraints: zero budget, local compute only, no paid APIs/cloud GPUs.
Keep all project downloads, caches, temporary files, and artifacts on D: using
`scripts/environment.ps1`. Reduce workloads if local training is constrained.

## Reproduction

```powershell
. .\scripts\environment.ps1
uv sync --locked --extra ml
uv run --locked --extra ml python scripts/phase0_smoke.py `
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 `
  --dataset-revision 1846984189f2bc0ac51c6fc3a600fa9e1714e0c6
```

The smoke runner resolves immutable upstream revisions, records license metadata,
loads only the official training partition, measures full-label prompt lengths,
checks the chat-template completion boundary, and masks prompt tokens from loss.
It uses rank-8 LoRA, BF16 when supported (otherwise scaled FP16), micro-batch 1,
16 accumulation steps, gradient checkpointing, and 20 optimizer updates. It never
truncates examples. This is a feasibility probe on 32 examples, not a benchmark.

Generated reports, canonical labels, and adapter files stay in ignored
`outputs/phase0/`. A failed run writes its traceback to `report.json`; use a new
`--output outputs/phase0-<attempt>` for each subsequent attempt to retain evidence.
Successful completion must include adapter reload and deterministic generation.
No official test evaluation, full training, or serving implementation is included.

The completed fixed-shape run is under `outputs/phase0-fixed-shape/`; the initial
run is under `outputs/phase0-attempt3/`. Canonical label IDs are array indices in
`reports/phase0-labels.json`. `configs/phase0.yaml` records source revisions and licenses.
All 18 existing tests and Ruff checks passed after runner changes. An upstream
Starlette TestClient deprecation warning remains; it does not fail tests.

Package source: https://pytorch.org/get-started/previous-versions/ (CUDA 12.8).
