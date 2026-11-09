"""Invalid outputs remain incorrect in headline metrics."""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    targets: Sequence[str], predictions: Sequence[str | None], labels: Sequence[str]
) -> dict[str, float]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Require equally sized, nonempty targets and predictions")
    if not labels or len(set(labels)) != len(labels) or any(t not in labels for t in targets):
        raise ValueError("Targets must belong to unique canonical labels")
    invalid = "__INVALID__"
    while invalid in labels:
        invalid += "_"
    normalized = [p if p in labels else invalid for p in predictions]
    return {
        "accuracy": float(accuracy_score(targets, normalized)),
        "macro_f1": float(
            f1_score(targets, normalized, labels=list(labels), average="macro", zero_division=0)
        ),
        "label_validity": sum(p in labels for p in predictions) / len(predictions),
    }
