import pytest

from domaintune.train import accumulation_size, encode_completion


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        if add_generation_prompt:
            return [1, 2, 3]
        assert messages[-1] == {"role": "assistant", "content": '{"intent": "card_arrival"}'}
        return [1, 2, 3, 4, 5]


def test_prompt_and_padding_never_contribute_to_loss():
    result = encode_completion(
        Tokenizer(),
        {"text": "Where is it?", "intent_name": "card_arrival"},
        ["card_arrival"],
        7,
    )
    assert result == {
        "input_ids": [1, 2, 3, 4, 5, 0, 0],
        "attention_mask": [1, 1, 1, 1, 1, 0, 0],
        "labels": [-100, -100, -100, 4, 5, -100, -100],
    }


def test_training_never_truncates():
    with pytest.raises(ValueError, match="truncation forbidden"):
        encode_completion(
            Tokenizer(), {"text": "hi", "intent_name": "card_arrival"}, ["card_arrival"], 4
        )


def test_partial_accumulation_uses_actual_size():
    assert [accumulation_size(i, 5, 3) for i in range(5)] == [3, 3, 3, 2, 2]


def test_misaligned_chat_template_rejected():
    class Broken(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
            return [1, 2] if add_generation_prompt else [9, 2, 3]

    with pytest.raises(ValueError, match="strict token prefix"):
        encode_completion(
            Broken(), {"text": "hi", "intent_name": "card_arrival"}, ["card_arrival"], 8
        )
