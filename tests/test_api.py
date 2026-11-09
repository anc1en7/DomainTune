import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_and_unready_model():
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.get("/model/info").status_code == 503
    assert client.post("/predict", json={"text": "My card was declined"}).status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 2001},
        {"text": 5},
        {"text": "hello", "extra": True},
    ],
)
def test_schema_rejects_invalid_inputs(payload):
    assert client.post("/predict", json=payload).status_code == 422
