"""
tests/test_api.py
End-to-end FastAPI integration tests using TestClient. Skipped automatically
if no model artifacts exist (so the suite passes on a fresh checkout before
the user has trained anything).
"""

import os
from pathlib import Path

import pytest

MODEL_PATH = "models/xgb/v1/model.json"
pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="No trained model. Run `make train` first.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "xgb" in r.json()["models_loaded"]


def test_predict_obvious_phish(client):
    r = client.post("/predict", json={
        "url": "http://paypa1-secure-verify-login.tk/wp-login.php?action=verify"
    })
    assert r.status_code == 200
    body = r.json()
    assert body["is_phish"] is True
    assert body["phish_probability"] > 0.7
    assert "request_id" in body


def test_predict_obvious_benign(client):
    r = client.post("/predict", json={"url": "https://www.google.com/"})
    assert r.status_code == 200
    assert r.json()["is_phish"] is False


def test_predict_with_explanation(client):
    r = client.post("/predict", json={
        "url": "http://login.paypal.evil-site.tk/verify",
        "explain": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"] is not None
    assert len(body["explanation"]) > 0
    assert "feature" in body["explanation"][0]
    assert "shap_value" in body["explanation"][0]


def test_batch_endpoint(client):
    urls = ["http://google.com",
            "http://login-verify.tk/wp-admin",
            "https://github.com/",
            "http://192.168.1.1/admin"]
    r = client.post("/predict/batch", json={"urls": urls})
    assert r.status_code == 200
    assert r.json()["count"] == 4


def test_feedback_round_trip(client):
    r = client.post("/predict", json={"url": "https://example.com/"})
    rid = r.json()["request_id"]
    fb = client.post("/feedback", json={
        "request_id": rid, "actual_label": 0,
    })
    assert fb.status_code == 200


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "phish_predictions_total" in r.text
