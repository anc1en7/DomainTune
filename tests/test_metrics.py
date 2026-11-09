import pytest

from domaintune.metrics import classification_metrics


def test_invalid_predictions_count_as_incorrect():
    result = classification_metrics(["a", "b"], ["a", None], ["a", "b"])
    assert result == {"accuracy": 0.5, "macro_f1": 0.5, "label_validity": 0.5}


def test_no_silent_length_mismatch():
    with pytest.raises(ValueError):
        classification_metrics(["a"], [], ["a"])
