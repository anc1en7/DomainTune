# Experiment log

Earlier experiments are recorded in the Phase 0 and baseline reports and session
handoff. Log new attempts here; preserve configurations, revisions, resources,
metrics, and artifact paths.

## Phase 3 feasibility

Two BF16 rank-8 LoRA optimizer updates at fixed length 640, accumulation 16,
on shuffled real training examples completed in 22.24 seconds. Peak allocated
2.07 GiB; reserved 2.29 GiB. All 8,992 train rows checked, lengths 530–625.
Completion-only loss and prompt/padding masking verified. See
`reports/phase3-feasibility.json` for config, revisions, environment, losses,
and source commit caveat in `reports/training.md`. Adapter and raw run artifacts
remain ignored at `outputs/adapters/phase3-feasibility/`.

## Phase 3 primary run: stopped at user request

Started with committed source `67bfb64`; user requested stopping for later
continuation. Saved epoch 1 after 562 optimizer updates covering all 8,992 training
rows. Mean update loss: 0.13799548. Retained adapter/tokenizer:
`outputs/adapters/phase3-r8/epoch-1`. All 96 adapter tensors are finite and the
tokenizer loads independently on CPU. Training and worker processes exited at
approximately 21:45 IST on 2026-11-10; GPU memory returned to 0 MiB.
Partial validation contains 103/1000 rows and is not a completed evaluation.
No checkpoint selection or test evaluation occurred. Structured evidence is in
`reports/phase3-stopped.json`. Optimizer/scheduler/RNG states were not saved;
exact continuation is unavailable. Preserve this limitation in any resumed
experiment and wait for the user's authorization before running more training.
