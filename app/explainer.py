"""
app/explainer.py
SHAP-based explanations for XGBoost predictions. Returns the top contributing
features so an analyst can see why a URL was flagged.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

log = logging.getLogger(__name__)


class XGBExplainer:
    def __init__(self, model: xgb.XGBClassifier):
        self.model = model
        # TreeExplainer is fast and exact for gradient-boosted trees
        self.explainer = shap.TreeExplainer(model)
        log.info("Initialized SHAP TreeExplainer")

    def explain(self, X: pd.DataFrame, top_k: int = 8
                ) -> list[list[dict]]:
        """Return per-row top-k feature contributions sorted by |shap_value|."""
        shap_values = self.explainer.shap_values(X)
        # XGBoost binary -> single matrix; multiclass -> list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        results = []
        cols = X.columns.tolist()
        for i in range(len(X)):
            row_shap = shap_values[i]
            row_vals = X.iloc[i].values
            order = np.argsort(np.abs(row_shap))[::-1][:top_k]
            results.append([
                {
                    "feature": cols[j],
                    "value": float(row_vals[j]),
                    "shap_value": float(row_shap[j]),
                }
                for j in order
            ])
        return results
