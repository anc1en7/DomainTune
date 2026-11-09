import pytest

from domaintune.parse import InvalidPrediction, parse_prediction


def test_canonical_output():
    assert parse_prediction('{"intent":"a"}', ["a", "b"]) == "a"


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '```json\n{"intent":"a"}\n```',
        '{"intent":"unknown"}',
        '{"intent":"a","reason":"x"}',
        '["a"]',
        '{"intent":["a"]}',
        '{"intent":"a","intent":"b"}',
    ],
)
def test_rejects_invalid_outputs(output):
    with pytest.raises(InvalidPrediction):
        parse_prediction(output, ["a", "b"])
