import copy

import pytest

from domaintune.audit import near_duplicates
from domaintune.data import normalize, prepare


@pytest.fixture
def source():
    return {
        "train": [
            {"text": f"class {i} training example {j}", "label": i}
            for i in range(77)
            for j in range(12)
        ],
        "test": [{"text": f"class {i} heldout", "label": i} for i in range(77)],
    }


def run(source):
    return prepare(source, [f"label_{i}" for i in range(77)], "a" * 40)


def test_reproducible_isolated_and_traceable(source):
    original = copy.deepcopy(source)
    splits, report = run(source)
    assert (splits, report) == run(source)
    assert source == original
    assert len({r["sample_id"] for rows in splits.values() for r in rows}) == 1001
    for split in ("train", "validation", "test"):
        assert len({r["intent_id"] for r in splits[split]}) == 77
        for row in splits[split]:
            assert source[row["source_split"]][row["source_index"]]["text"] == row["text"]
    hashes = {
        s: {r["normalized_sha256"] for r in splits[s]} for s in ("train", "validation", "test")
    }
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["validation"] & hashes["test"]
    assert [r["text"] for r in splits["test"]] == [r["text"] for r in source["test"]]


def test_duplicate_conflict_and_test_quarantine(source):
    source["train"] += [
        {"text": " CLASS 0 HELDOUT  ", "label": 1},
        dict(source["train"][0]),
        {"text": source["train"][1]["text"], "label": 1},
    ]
    splits, report = run(source)
    assert report["exclusion_reasons"] == {
        "official_test_text_overlap": 1,
        "duplicate_training_text": 1,
        "conflicting_training_labels": 2,
    }
    assert len(splits["test"]) == 77


@pytest.mark.parametrize("value", [None, "", " \t", "\u00a0"])
def test_empty_text_rejected(source, value):
    source["train"][0]["text"] = value
    with pytest.raises(ValueError, match="text"):
        run(source)


@pytest.mark.parametrize("label", [-1, 77, True, "0"])
def test_invalid_label_rejected(source, label):
    source["train"][0]["label"] = label
    with pytest.raises(ValueError, match="label"):
        run(source)


def test_normalization():
    assert normalize("  ＢＡＮＫ\u00a0PayMent\n") == "bank payment"


def test_near_duplicate_cross_split_candidates():
    def row(i, text):
        return {"sample_id": str(i), "text": text, "intent_id": 0}

    splits = {
        "train": [row(1, "my card was lost yesterday")],
        "validation": [row(2, "my card was lost yesterday!")],
        "test": [row(3, "please cancel this bank transfer")],
    }
    pairs = near_duplicates(splits, threshold=0.8)
    assert len(pairs) == 1
    assert (pairs[0]["left_id"], pairs[0]["right_id"]) == ("1", "2")
