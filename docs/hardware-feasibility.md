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

PyTorch build, CUDA runtime, BF16 support, model loading,
token counts, peak memory, and 20-step training feasibility remain unverified.
Driver-reported CUDA capability is not proof of a compatible installed PyTorch runtime.
The 4 GiB GPU requires an actual measured LoRA smoke test before choosing settings.
User constraints: zero budget, local compute only, no paid APIs/cloud GPUs.
Keep all project downloads, caches, temporary files, and artifacts on D: using
`scripts/environment.ps1`. Reduce workloads if local training is constrained.
