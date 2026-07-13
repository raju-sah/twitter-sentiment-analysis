import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_positive():
    r = client.post("/predict", json={"text": "I love this, it is great!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["processed_tokens"], list)


def test_predict_negative():
    r = client.post("/predict", json={"text": "This is terrible and I hate it."})
    assert r.status_code == 200
    assert r.json()["label"] == "negative"


def test_empty_text():
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 400
