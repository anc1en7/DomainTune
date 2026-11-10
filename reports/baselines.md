# Phase 2 validation baselines

All systems use the frozen Phase 1 partitions: 8,992 training rows and 1,000
validation rows. The official test partition is not loaded by the runner. The
near-duplicate risks described in `data_audit.md` still apply. These are validation
results used for selection, not unbiased final test estimates.

Completed locally on 2026-11-10. No official test evaluation or full training ran.

| System | Validation accuracy | Macro-F1 | Invalid output |
|---|---:|---:|---:|
| Majority class | 1.9% | 0.000484 | 0% |
| TF-IDF + logistic regression | 89.2% | 0.890987 | 0% |
| Unchanged Qwen 0.5B, v1 zero-shot | 13.7% | 0.152461 | 60.8% |

The unchanged model produced parseable JSON for 99.9% of messages, but only 39.2%
met the strict output contract. There were 137 correct, 255 valid but wrong, and
608 invalid predictions. Formatting alone does not ensure a canonical intent.
This full-label zero-shot configuration is substantially worse than the selected
TF-IDF baseline. No claim is made about other prompts or models. Keep this v1
prompt fixed for the primary LoRA comparison; any future prompt experiment must
be separately versioned and evaluated on validation only.

Qwen latency was 1,022 ms P50 and 1,441 ms P95; serial throughput was 0.950
messages/second. Peak allocated GPU memory was 1.002 GiB and peak reserved memory
1.109 GiB. Timed inference totaled about 17.5 minutes, excluding model loading
and warm-up. These measurements are subject to the mixed-load caveat below.

## Reproduce

Run from the repository root with the locked ML extra installed:

```powershell
. .\scripts\environment.ps1
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
# B0 and B1 (same command, training and selection take place locally)
.\.venv\Scripts\python.exe scripts/run_baselines.py --config configs/baseline.yaml --system classical
# B2 (cached pinned base weights, no adapter)
.\.venv\Scripts\python.exe scripts/run_baselines.py --config configs/baseline.yaml --system llm
```

Omit `--system` to run all three together. Cached prepared data and pinned model
weights from Phases 0/1 are prerequisites. Retrieval instructions and revisions
are recorded in the hardware notes and data audit. Outputs are written to
`reports/baselines-classical.json`, `reports/baselines-llm.json`, or
`reports/baselines-all.json` respectively. Generation logs stay ignored under
`outputs/phase2/` on D:. Rerunning replaces the corresponding run outputs.

Before fitting or prediction, the runner verifies processed train and validation
SHA-256 values against frozen provenance, then checks text-free row manifests and
label mappings. Reports record those hashes, the provenance hash, prompt source
hash, lockfile hash, source commit, configuration, and framework versions.
The raw generation log remains local; the committed report includes the wrong
sample IDs and predictions, not dataset text or model weights.

## Classical baselines

The majority class is determined from training counts; canonical label order
breaks ties. It selects `card_payment_fee_charged`. No validation labels influence
this choice. All 77 labels remain in macro-F1, including labels never predicted.

TF-IDF is fitted separately for each candidate on training text only. Candidate
settings were declared before results: word 1-2 grams or character-within-word
3-5 grams, `C` in {1, 4}; `min_df=2`, `sublinear_tf=True`, other vectorizer defaults.
Logistic regression uses lbfgs, 1,000 maximum iterations, seed 42, and no class
weighting. No convergence warnings occurred. The audit vectorizer is never used.
Choose the highest validation macro-F1, with first candidate winning exact ties.

| TF-IDF analyzer | C | Validation macro-F1 |
|---|---:|---:|
| Word | 1 | 0.855950 |
| Word | 4 | 0.872385 |
| Character within word | 1 | 0.858067 |
| Character within word | 4 | 0.890987 |

Selected settings are character-within-word 3-5 grams and `C=4`. Selection is based
on this one validation partition; the score is subject to selection optimism.
No classical model weights are distributed: the command deterministically refits
the selected model from frozen training data. Per-class metrics, confusion counts,
top 15 confusions, and wrong sample IDs are recorded in the JSON report.

The largest individual confusion counts are three each: reverted card payments
predicted as declined payments, transfer fees predicted as bank top-up fees, and
card activation predicted as card arrival. The lowest validation class F1 is
`order_physical_card` (0.621), followed by `card_not_working` and
`virtual_card_not_working` (both 0.667). These estimates have small class supports
and are descriptive; no test results or label corrections inform them.

## Unchanged instruction model protocol

Use Qwen/Qwen2.5-0.5B-Instruct at
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16 on the local RTX 3050 Laptop GPU,
batch size 1, SDPA attention, seed 42. No adapter is loaded. Use the existing v1
full-label prompt and the official chat template. Decoding is greedy
(`do_sample=False`), capped at 48 new tokens, with EOS as the padding ID. Model
generation-config sampling fields trigger an informational Transformers warning
but are inactive during greedy decoding. No prompt search, few-shot examples,
constrained decoding, label repair, or fuzzy matching is applied.

Strict JSON must contain exactly one `intent` field and a canonical label.
Duplicate keys, unknown labels, extra text, or malformed output are invalid and
incorrect. JSON parse rate is reported separately from strict label validity:
parseable JSON can still violate the output contract. Invalid predictions receive
their own confusion column and reduce recall in their true class; headline
macro-F1 averages over the same 77 canonical classes for every system.

No inputs are truncated. Inputs that exceed model context fail explicitly.
Generation that exhausts its output budget is evaluated as returned, with the same
parser. Three untimed validation examples warm the model; each of the 1,000 rows
is then timed once, in source order. Timing includes chat tokenization, device
transfer, generation, and decoding, with CUDA synchronization. P50/P95 and
throughput describe this local serial run, not a production load test. Peak GPU
allocated/reserved memory is reset after warm-up and measured over evaluation.
The recorded Phase 2 run initially overlapped with classical-baseline CPU work;
latency is descriptive under that mixed system load, not an isolated performance
comparison. A controlled benchmark across selected models remains future work.

## Verification

The evaluation tests check a hand-calculated case, including an invalid prediction,
per-class support and confusion counts. Existing metric tests cover invalids and
length mismatches. A checksum-tampering test fails before modeling, and a fit spy
verifies that validation text never enters TF-IDF fitting. The runner does not
open processed test rows. Final official test evaluation belongs to Phase 5.
A second full classical run reproduced all candidate metrics and selected-model
per-class scores, error IDs, and confusion counts exactly; only timings and source
commit metadata changed.
All 1,000 saved Qwen outputs were independently reparsed after completion and
matched to frozen validation sample IDs. Recomputed aggregate metrics, per-class
metrics, confusion matrix, and error records match the committed report exactly.
All 33 tests and Ruff lint/format checks passed. B0/B1/B2 are reproducible using
the commands above. Phase 2 is complete; stop before Phase 3 until requested.
