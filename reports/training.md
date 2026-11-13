# Phase 3: primary LoRA experiment

Status: reset-state continuation authorized on 2026-11-13; Phase 3 incomplete.

Epoch 1 completed 562 optimizer updates over all 8,992 training rows; mean update
loss 0.13799548. The adapter and tokenizer are retained under
`outputs/adapters/phase3-r8/epoch-1`. CPU integrity checks read all 96 finite
adapter tensors and loaded the saved tokenizer. GPU training and worker processes
have exited. Validation stopped after 103/1000 rows; no full validation metric or
checkpoint selection is available. No official test evaluation ran.

See `reports/phase3-stopped.json` and `docs/session-handoff.md`. Optimizer,
scheduler, and RNG states were not saved: exact continuation is unavailable.
Any later continuation from this adapter must be labeled as a separate experiment
with reset state, or the original experiment can be restarted in a new directory.
Wait for the user's instruction before resuming. Add stateful checkpoints and
graceful stopping before another long run.

Continuation work on 2026-11-13: the evaluation CLI now accepts a fresh `--output`
directory for standalone validation of the retained epoch-1 checkpoint. It preserves
the interrupted predictions and does not claim reproduction when no completed
training-time validation exists. Example:

```powershell
. .\scripts\environment.ps1
.venv\Scripts\python.exe scripts/evaluate_model.py --checkpoint outputs/adapters/phase3-r8/epoch-1 --output outputs/phase3-epoch1-validation
```

The user chose reset-state continuation after epoch 1 on 2026-11-13. Two additional
epochs use `configs/lora_r8_continuation.yaml`, with epoch numbers 2 and 3 and a
fresh 1,124-update linear schedule with 3% warmup. This is a separate experiment,
not an exact resume. The original run and partial validation remain untouched.
Full standalone epoch-1 validation supplies the initial checkpoint candidate.

The runner saves full training state every 100 updates, before/after validation,
and on a graceful stop. A `STOP` file inside the active output directory requests
a stop at the next optimizer update or validation example. A fresh output can
restore adapter, optimizer, scheduler, Python/NumPy/PyTorch/CUDA RNG states, and
data position using `--resume-state <path-to-training-state.pt>` with the same
config. The stop/resume probe compares three uninterrupted GPU updates with one
update followed by restoration and two further updates. Probe artifacts stay
ignored under `outputs/adapters/continuation-check-*`.

The initial three-update probe showed a maximum loss difference of 0.000313
and maximum adapter-element difference of 0.000368 after restoration. This is
not evidence of bitwise-identical GPU continuation. A separate restoration audit
checks every loaded optimizer tensor and the scheduler before any resumed
update. The original epoch-1 continuation still explicitly resets optimizer and
scheduler state. Probe source commits precede the implementation commit because
the checks ran while the implementation was being verified.

The 640-token probe subsequently passed two optimizer updates in 22.24 seconds:
peak allocated 2,222,181,376 bytes (2.07 GiB), reserved 2,462,056,448 bytes
(2.29 GiB). All 8,992 rows fit, with actual lengths 530–625. Probe losses were
0.49890 and 0.60047; this proves mechanics and memory feasibility, not quality.
The probe uses a two-update scheduler and is separate from primary training.
Its report records the pre-implementation commit because the probe ran before
the implementation milestone was committed; primary training records its exact
committed implementation. The retained evidence is `reports/phase3-feasibility.json`.

The primary experiment uses `configs/lora_r8.yaml`: the pinned Qwen model,
version-1 full-label chat prompt, BF16 SDPA, rank 8 on q_proj/v_proj, fixed
640-token padding, microbatch 1, accumulation 16, AdamW, linear learning-rate
decay with 3% warmup, gradient norm clipping at 1.0, and three epochs.
The last partial accumulation update is normalized by its actual row count.
Only assistant completion tokens (including the chat terminator) contribute to
loss. Neither prompts nor padding do. No truncation is permitted.

All 8,992 frozen training rows are tokenized and checked before model loading.
The runner verifies frozen train/validation hashes and manifests and never loads
processed official test rows. Each epoch saves an adapter and tokenizer and runs
greedy validation with the matched 48-token generation budget and strict parser.
Invalid outputs count as incorrect. Highest validation macro-F1 selects the
checkpoint; ties retain the earlier epoch. No prompt or hyperparameter search is
part of this primary run.

Run in PowerShell from the repository:

```powershell
. .\scripts\environment.ps1
.venv\Scripts\python.exe scripts/train_peft.py --output outputs/adapters/phase3-feasibility --probe-steps 2
.venv\Scripts\python.exe scripts/train_peft.py --config configs/lora_r8.yaml
# Substitute the selected checkpoint recorded in run.json:
.venv\Scripts\python.exe scripts/evaluate_model.py --checkpoint outputs/adapters/phase3-r8/epoch-N --split validation
```

Use a fresh output directory for each attempt; overwriting evidence is rejected.
Everything runs from existing local artifacts without paid services. Adapters,
raw outputs, training logs, and intermediate reports remain ignored on D: under
`outputs/adapters/`. `run.json` records revisions, source commit, configuration,
package versions, data/prompt/lock hashes, loss, timing, and peak allocated/reserved
GPU memory. `training.jsonl` records every optimizer update. Failure reports retain
the traceback and available GPU memory peaks. The reload command starts a fresh
process and requires exact agreement with the selected epoch's validation metrics,
confusion matrix, and error IDs.

Test evaluation is intentionally unavailable in this Phase 3 CLI. It belongs to
Phase 5, after the ablation and final model selection in Phase 4.
