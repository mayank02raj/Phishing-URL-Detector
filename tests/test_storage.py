"""tests/test_storage.py"""

import tempfile

import pytest

from app.storage import PredictionStore


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        s = PredictionStore(f.name)
        yield s


def test_log_and_recall(store):
    rid = store.log_prediction(
        url="http://test.tk/login",
        model="xgb",
        probability=0.85,
        is_phish=True,
        threshold=0.5,
        latency_ms=1.2,
        features={"url_length": 22},
    )
    assert rid is not None
    stats = store.stats()
    assert stats["total_predictions"] == 1
    assert stats["phish_predictions"] == 1


def test_feedback_round_trip(store):
    rid = store.log_prediction(
        "http://test.com", "xgb", 0.3, False, 0.5, 1.0, {"url_length": 14})
    assert store.record_feedback(rid, actual_label=1) is True
    assert store.record_feedback("nonexistent-id", actual_label=1) is False


def test_recent_features_empty_when_no_data(store):
    df = store.recent_features(10)
    assert df.empty


def test_recent_features_returns_dataframe(store):
    for i in range(5):
        store.log_prediction(
            f"http://test{i}.com", "xgb", 0.2 + i * 0.1,
            False, 0.5, 1.0, {"url_length": 14 + i})
    df = store.recent_features(10)
    assert len(df) == 5
    assert "url_length" in df.columns
