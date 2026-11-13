import pytest

from domaintune.predict import EvaluationStopped, evaluate_adapter


def test_validation_stop_preserves_partial_log_without_reporting_metrics(tmp_path):
    pytest.importorskip("torch")

    class Model:
        def eval(self):
            pass

    destination = tmp_path / "partial.jsonl"
    with pytest.raises(EvaluationStopped, match="example boundary"):
        evaluate_adapter(
            Model(), None, [{"text": "unused"}], [], destination, stop_requested=lambda: True
        )
    assert destination.read_text("utf-8") == ""
