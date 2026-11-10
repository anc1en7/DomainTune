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

Model loading, token counts, peak memory, and 20-step training feasibility remain unverified.
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

Package source: https://pytorch.org/get-started/previous-versions/ (CUDA 12.8).
