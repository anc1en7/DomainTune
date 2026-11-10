"""Shared evaluation with an explicit invalid-output confusion column."""

import json
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support

from domaintune.metrics import classification_metrics


def evaluation_report(rows, predictions, labels, raw_outputs=None):
    targets = [r["intent_name"] for r in rows]
    metrics = classification_metrics(targets, predictions, labels)
    normalized = [p if p in labels else "__INVALID__" for p in predictions]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, normalized, labels=labels, zero_division=0
    )
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for target, prediction in zip(targets, normalized, strict=True):
        matrix[labels.index(target)][
            labels.index(prediction) if prediction in labels else len(labels)
        ] += 1
    result = {
        "count": len(rows),
        **metrics,
        "invalid_rate": 1 - metrics["label_validity"],
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion": {"rows": labels, "columns": [*labels, "__INVALID__"], "matrix": matrix},
        "errors": [
            {"sample_id": row["sample_id"], "target": target, "prediction": prediction}
            for row, target, prediction in zip(rows, targets, predictions, strict=True)
            if target != prediction
        ],
        "top_confusions": [
            {"target": a, "prediction": b, "count": n}
            for (a, b), n in Counter(
                (a, b) for a, b in zip(targets, normalized, strict=True) if a != b
            ).most_common(15)
        ],
    }
    if raw_outputs is not None:
        if len(raw_outputs) != len(rows):
            raise ValueError("Output count mismatch")
        valid_json = 0
        for output in raw_outputs:
            try:
                json.loads(output)
                valid_json += 1
            except (ValueError, TypeError):
                pass
        result["json_parse_rate"] = valid_json / len(rows)
    return result
