"""
app/drift.py
Population Stability Index (PSI) drift detection.

PSI measures how much a feature's distribution has shifted from a reference
window (training time) to a recent window (production traffic). Conventional
thresholds:
    PSI < 0.1   stable
    0.1 - 0.2   small shift, watch
    PSI > 0.2   significant drift, retrain candidate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def psi(reference: np.ndarray, recent: np.ndarray, n_bins: int = 10) -> float:
    """Compute PSI between two 1D distributions."""
    if len(reference) == 0 or len(recent) == 0:
        return 0.0
    edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_hist, _ = np.histogram(reference, bins=edges)
    rec_hist, _ = np.histogram(recent, bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    rec_pct = rec_hist / max(rec_hist.sum(), 1)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    rec_pct = np.where(rec_pct == 0, 1e-6, rec_pct)

    return float(np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct)))


class DriftMonitor:
    def __init__(self, reference_stats_path: str, threshold: float = 0.2):
        self.threshold = threshold
        self.reference: dict | None = None
        path = Path(reference_stats_path)
        if path.exists():
            with open(path) as f:
                self.reference = json.load(f)
            log.info("Loaded reference stats from %s", path)
        else:
            log.warning("No reference stats at %s, drift disabled", path)

    def compute_drift(self, recent_features: pd.DataFrame
                      ) -> tuple[dict[str, float], list[str], str]:
        if self.reference is None or recent_features.empty:
            return {}, [], "ok"

        # Use the quantiles in reference_stats.json to estimate the
        # reference distribution per feature with a synthetic sample.
        scores = {}
        for col in recent_features.columns:
            ref_q = self.reference["feature_quantiles"].get(col)
            if not ref_q:
                continue
            # Build a synthetic ref distribution from quantiles
            q10, q50, q90 = ref_q
            synthetic_ref = np.concatenate([
                np.full(100, q10),
                np.full(100, q50),
                np.full(100, q90),
            ])
            scores[col] = round(
                psi(synthetic_ref, recent_features[col].values), 4)

        drifted = [k for k, v in scores.items() if v > self.threshold]
        if len(drifted) >= 5:
            status = "drifted"
        elif drifted:
            status = "warning"
        else:
            status = "ok"
        return scores, drifted, status
