import json

import pytest

from domaintune.baselines import load_frozen_data
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
