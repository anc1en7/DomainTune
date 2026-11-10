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
