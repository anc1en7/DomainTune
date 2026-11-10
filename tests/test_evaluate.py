import json

import pytest

from domaintune.baselines import classical_baselines, load_frozen_data
from domaintune.evaluate import evaluation_report


def test_invalid_column_and_hand_computed_metrics():
    rows = [
        {"sample_id": str(i), "intent_name": target}
        for i, target in enumerate(["a", "a", "b", "b"])
    ]
    report = evaluation_report(
        rows, ["a", "b", "b", None], ["a", "b"], ['{"intent":"a"}', "{}", "[]", "bad"]
    )
    assert report["accuracy"] == 0.5
    assert report["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert report["confusion"]["matrix"] == [[1, 1, 0], [0, 1, 1]]
    assert report["invalid_rate"] == 0.25
    assert report["json_parse_rate"] == 0.75
    assert len(report["errors"]) == 2
    assert report["per_class"]["b"]["support"] == 2


def test_tampered_data_rejected_before_modeling(tmp_path):
    (tmp_path / "data/manifests").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir()
    (tmp_path / "data/manifests/provenance.json").write_text(
        json.dumps({"labels": ["a"], "processed_sha256": {"train": "wrong"}})
    )
    (tmp_path / "data/processed/train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_data(tmp_path)


def test_vectorizer_fits_training_only(monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer

    fitted = []
    original = TfidfVectorizer.fit_transform

    def tracked_fit(self, corpus, y=None):
        fitted.append(list(corpus))
        return original(self, corpus, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", tracked_fit)
    train = [
        {"text": text, "intent_name": label}
        for text, label in [
            ("card missing", "a"),
            ("card missing now", "a"),
            ("transfer pending", "b"),
            ("transfer pending now", "b"),
        ]
    ]
    validation = [{"sample_id": "heldout", "text": "unseen validation", "intent_name": "a"}]
    result = classical_baselines(
        {"train": train, "validation": validation},
        ["a", "b"],
        {"seed": 42, "tfidf": {"analyzers": ["word"], "C": [1.0]}},
    )
    assert fitted == [[r["text"] for r in train]]
    assert result["majority"]["selected_label"] == "a"
