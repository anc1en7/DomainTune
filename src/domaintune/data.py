"""Deterministic, text-grouped BANKING77 preparation; test rows stay intact."""

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text):
    """Audit key only: preserve original input for modeling."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare(raw, labels, revision, seed=42, validation_fraction=0.1):
    if len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Expected 77 unique canonical labels")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    records = []
    for split in ("train", "test"):
        for index, row in enumerate(raw[split]):
            text, label = row["text"], row["label"]
            if not isinstance(text, str) or not normalize(text):
                raise ValueError(f"Empty/invalid text at {split}:{index}")
            if type(label) is not int or not 0 <= label < len(labels):
                raise ValueError(f"Invalid label at {split}:{index}")
            records.append(
                {
                    "sample_id": "sha256:"
                    + digest(canonical_json([revision, split, index, text, label])),
                    "text": text,
                    "intent_id": label,
                    "intent_name": labels[label],
                    "source_split": split,
                    "source_index": index,
                    "dataset_revision": revision,
                    "text_sha256": digest(text),
                    "normalized_sha256": digest(normalize(text)),
                }
            )
    groups = defaultdict(list)
    for row in records:
        groups[row["normalized_sha256"]].append(row)
    eligible, excluded = [], []
    test = [r for r in records if r["source_split"] == "test"]
    for key in sorted(groups):
        group = groups[key]
        training = [r for r in group if r["source_split"] == "train"]
        reason = None
        if any(r["source_split"] == "test" for r in group):
            reason = "official_test_text_overlap"
        elif len({r["intent_id"] for r in training}) > 1:
            reason = "conflicting_training_labels"
        if reason:
            excluded.extend(dict(r, exclusion_reason=reason) for r in training)
        elif training:
            eligible.append(training[0])
            excluded.extend(
                dict(r, exclusion_reason="duplicate_training_text") for r in training[1:]
            )
    train, validation = train_test_split(
        eligible,
        test_size=validation_fraction,
        random_state=seed,
        stratify=[r["intent_id"] for r in eligible],
    )
    splits = {"train": train, "validation": validation, "test": test, "excluded": excluded}
    for rows in splits.values():
        rows.sort(key=lambda r: (r["source_split"], r["source_index"]))
    counts = {
        s: dict(sorted(Counter(r["intent_name"] for r in rows).items()))
        for s, rows in splits.items()
    }
    for s in ("train", "validation", "test"):
        if len(counts[s]) != 77:
            raise ValueError(f"Missing classes in {s}")
    report = {
        "source_rows": {s: len(raw[s]) for s in ("train", "test")},
        "source_sha256": digest(canonical_json(records)),
        "split_rows": {s: len(rows) for s, rows in splits.items()},
        "class_counts": counts,
        "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in excluded)),
        "unicode": {
            "non_ascii": sum(not r["text"].isascii() for r in records),
            "nfkc_changed": sum(
                unicodedata.normalize("NFKC", r["text"]) != r["text"] for r in records
            ),
            "replacement_character": sum("\ufffd" in r["text"] for r in records),
            "control_characters": sum(
                any(unicodedata.category(c) == "Cc" for c in r["text"]) for r in records
            ),
        },
        "character_lengths": {
            s: {
                "min": min(len(r["text"]) for r in rows),
                "max": max(len(r["text"]) for r in rows),
                "mean": sum(len(r["text"]) for r in rows) / len(rows),
            }
            for s, rows in splits.items()
            if rows
        },
        "duplicate_groups": {
            "exact": sum(n > 1 for n in Counter(r["text_sha256"] for r in records).values()),
            "normalized": sum(len(g) > 1 for g in groups.values()),
            "conflicting": sum(len({r["intent_id"] for r in g}) > 1 for g in groups.values()),
        },
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {r["normalized_sha256"] for r in splits[a]} & {
            r["normalized_sha256"] for r in splits[b]
        }:
            raise ValueError(f"Prohibited overlap: {a}/{b}")
    return splits, report
