# Phase 1 data audit

Completed locally on 2026-11-10. No models trained or evaluated in this phase.

## Source and reproduction

BANKING77: PolyAI/banking77, revision
`1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, CC-BY-4.0 (upstream metadata).
Attribution: Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence
Encoders. The existing Phase 0 canonical label array is verified byte-for-value
against the loaded feature mapping; indices remain the original 77 intent IDs.
The local cached build is checked for the pinned revision, including offline mode.

```powershell
. .\scripts\environment.ps1
$env:HF_HUB_OFFLINE='1' # omit these two offline flags on first download
$env:HF_DATASETS_OFFLINE='1'
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/data.yaml
```

Requires the locked ML extra. Metadata and text-free row manifests are committed
under `data/manifests/`; original texts and processed rows stay ignored on D:.
`provenance.json` records source and processed SHA-256 hashes, exact labels,
package versions, class counts, lengths, configuration, and near-duplicate IDs.
Sample IDs hash revision, source split, source index, original text, and label;
separate raw and normalized text hashes support overlap checks.

## Policy frozen before baselines

Normalization is NFKC, casefold, and whitespace collapse, used only for audit keys.
Original model inputs are preserved. Exclude every official training row matching
normalized test text, without consulting test labels to decide exclusion. Retain
all official test records, in order. Within training, quarantine all members of a
normalized group with conflicting labels; otherwise retain its first source row.
Split the remaining unique training rows with stratification, validation fraction
0.1, seed 42. Learned model preprocessing must fit only the resulting train split.

The official test is accessed here solely for provenance and leakage audit. No
test predictions, prompt selection, hyperparameter selection, or relabeling occur.
These cleaned training partitions differ from using the full official training
partition; future comparisons must use these same manifests and disclose that.

## Measurements

| Partition | Rows | Per-class minimum | Per-class maximum | Character lengths |
|---|---:|---:|---:|---|
| Official train | 10,003 | See provenance | See provenance | Before exclusions |
| Prepared train | 8,992 | 32 | 168 | 13–433, mean 59.50 |
| Validation | 1,000 | 3 | 19 | 15–422, mean 59.36 |
| Official test, unchanged | 3,080 | 40 | 40 | 13–368, mean 54.23 |
| Excluded training rows | 11 | — | — | 25–111 |

All 77 classes occur in each usable partition. There are zero byte-exact duplicate
groups, 12 normalized duplicate groups, and zero conflicting-label normalized
groups across the source. Seven training rows overlap normalized official test
text; four additional training duplicates are excluded. A normalized duplicate
within the official test is retained by policy. All train/validation/test pairwise
normalized-hash intersections are empty after exclusions.

All texts are nonempty and all label IDs valid. Source Unicode audit: 61 non-ASCII
rows, 3 NFKC-changing rows, 13 rows containing control characters, zero replacement
characters. Control characters include line breaks; no automatic text repair is
applied. The pipeline fails on empty normalized text or invalid labels.

## Near-duplicate investigation and remaining risk

Audit-only TF-IDF over normalized character 3–5 grams, cosine >= 0.90, yields 137
cross-partition candidate pairs: 17 train/validation, 110 train/test, and 10
validation/test. This vectorizer uses the corpus solely to measure similarities;
it must never be reused by the future modeling pipeline. Sparse products are
computed in 128-row blocks. This heuristic cannot guarantee semantic isolation.

Reviewed the flagged candidates: many differ only by punctuation, articles, or
short phrases; some long utterances are near verbatim. Retain near-duplicates
with an explicit residual leakage warning rather than silently changing the
official benchmark or treating a similarity threshold as verified equivalence.
Validation and eventual test performance may be optimistic on these examples.
Do not use test candidate contents to design prompts or select model settings.

Three candidates have different canonical labels. Reviewed ten training examples
per class for each involved pair (60 examples total), using the first ten retained
source-order records in each class:

- `why_verify_identity` / `verify_my_identity`: motivation versus verification
  procedure; some training examples mix these concepts.
- `exchange_via_app` / `fiat_currency_support`: performing conversion versus
  supported currencies; short exchange questions can omit the distinguishing cue.
- `pending_transfer` / `balance_not_updated_after_bank_transfer`: pending outgoing
  transfer versus a missing incoming balance update; short messages are ambiguous.

Preserve upstream labels in every case. No class merging or correction is made.
Candidate IDs and scores permit review without publishing dataset text.

## Verification and exit gate

Synthetic tests cover deterministic IDs and splits, preservation of source/test
order and content, normalized leakage isolation, duplicate/conflict quarantine,
invalid texts/labels, and near-duplicate detection. The full pinned pipeline was
rerun locally; its processed split hashes match the first run. A future run fails
before replacing rows if its processed hashes differ from frozen provenance.
All samples, including exclusions, are traceable by source index and revision.
Phase 1 exits with near-duplicate risk recorded; Phase 2 requires user approval.
