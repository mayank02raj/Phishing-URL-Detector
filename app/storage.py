"""
app/storage.py
SQLite prediction logging. Lightweight, no external dependency.

Schema:
    predictions(
        id, ts, request_id, url, model, probability, is_phish,
        threshold, latency_ms, features_json, actual_label, feedback_ts
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    model TEXT NOT NULL,
    probability REAL NOT NULL,
    is_phish INTEGER NOT NULL,
    threshold REAL NOT NULL,
    latency_ms REAL NOT NULL,
    features_json TEXT,
    actual_label INTEGER,
    feedback_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts
    ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id
    ON predictions(request_id);
"""


class PredictionStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
        log.info("PredictionStore ready at %s", db_path)

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def log_prediction(
        self,
        url: str,
        model: str,
        probability: float,
        is_phish: bool,
        threshold: float,
        latency_ms: float,
        features: dict | None = None,
    ) -> str:
        request_id = str(uuid.uuid4())
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO predictions
                   (ts, request_id, url, model, probability,
                    is_phish, threshold, latency_ms, features_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    request_id, url, model, probability,
                    int(is_phish), threshold, latency_ms,
                    json.dumps(features) if features else None,
                ),
            )
        return request_id

    def record_feedback(self, request_id: str, actual_label: int) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute(
                """UPDATE predictions
                   SET actual_label = ?, feedback_ts = ?
                   WHERE request_id = ?""",
                (actual_label, datetime.utcnow().isoformat(), request_id),
            )
            return cur.rowcount > 0

    def recent_features(self, n: int = 1000) -> pd.DataFrame:
        """Pull the last N predictions' features for drift monitoring."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT features_json FROM predictions
                   WHERE features_json IS NOT NULL
                   ORDER BY id DESC LIMIT ?""", (n,)).fetchall()
        if not rows:
            return pd.DataFrame()
        feats = [json.loads(r[0]) for r in rows]
        return pd.DataFrame(feats)

    def stats(self) -> dict:
        with self._conn() as c:
            row = c.execute("""
                SELECT COUNT(*),
                       SUM(is_phish),
                       AVG(latency_ms),
                       COUNT(actual_label)
                FROM predictions
            """).fetchone()
        total, phish, avg_lat, with_feedback = row
        return {
            "total_predictions": total or 0,
            "phish_predictions": phish or 0,
            "avg_latency_ms": round(avg_lat or 0, 2),
            "with_feedback": with_feedback or 0,
        }

    def health(self) -> str:
        try:
            with self._conn() as c:
                c.execute("SELECT 1")
            return "ok"
        except Exception as e:
            return f"error: {e}"
