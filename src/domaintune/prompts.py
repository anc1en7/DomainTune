"""Versioned prompt shared by unchanged and fine-tuned models."""

import json
from collections.abc import Sequence

PROMPT_VERSION = "v1"


def build_messages(text: str, labels: Sequence[str]) -> list[dict[str, str]]:
    if not text.strip() or not labels or len(set(labels)) != len(labels):
        raise ValueError("Require nonempty text and unique canonical labels")
    return [
        {
            "role": "system",
            "content": (
                "You classify banking support messages. Respond with ONLY valid JSON "
                'containing exactly one field: {"intent": "<allowed_label>"}. '
                "Do not invent a label or include explanations. Allowed labels: "
                + json.dumps(list(labels), ensure_ascii=False)
            ),
        },
        {"role": "user", "content": "Classify this support message:\n" + json.dumps(text)},
    ]
