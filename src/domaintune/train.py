"""Completion-only training helpers; no truncation or test-partition access."""

import json

from domaintune.prompts import build_messages


def encode_completion(tokenizer, row, labels, max_length):
    messages = build_messages(row["text"], labels)
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    target = json.dumps({"intent": row["intent_name"]})
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target}], tokenize=True
    )
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Assistant boundary must be a strict token prefix")
    if len(full) > max_length:
        raise ValueError(f"Sequence needs {len(full)} tokens; truncation forbidden")
    if row["intent_name"] not in labels:
        raise ValueError("Noncanonical training label")
    return {
        "input_ids": full + [tokenizer.pad_token_id] * (max_length - len(full)),
        "attention_mask": [1] * len(full) + [0] * (max_length - len(full)),
        "labels": [-100] * len(prefix) + full[len(prefix) :] + [-100] * (max_length - len(full)),
    }


def accumulation_size(index, count, accumulation):
    """Normalize the final partial update by its actual microbatch count."""
    start = (index // accumulation) * accumulation
    return min(accumulation, count - start)
