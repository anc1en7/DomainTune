"""Strict generation-output contract shared by evaluation and serving."""

import json
from collections.abc import Collection


class InvalidPrediction(ValueError):
    """The model did not produce exactly one canonical intent."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPrediction("Duplicate JSON field")
        result[key] = value
    return result


def parse_prediction(output: str, allowed_labels: Collection[str]) -> str:
    try:
        value = json.loads(output, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidPrediction("Output must be valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"intent"}:
        raise InvalidPrediction("Output must contain exactly the intent field")
    intent = value["intent"]
    if not isinstance(intent, str) or intent not in allowed_labels:
        raise InvalidPrediction("Unknown intent")
    return intent
