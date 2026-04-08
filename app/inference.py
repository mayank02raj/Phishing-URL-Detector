"""
app/inference.py
Loads XGBoost and CharCNN models, exposes a unified predict interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from app.config import settings
from ml.features import extract_features, feature_names
from ml.train_cnn import CharCNN, encode_url

log = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self):
        self.xgb_model: xgb.XGBClassifier | None = None
        self.cnn_model: CharCNN | None = None
        self.feature_names = feature_names()
        self.loaded: list[str] = []
        self._load()

    def _load(self):
        # XGBoost
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(settings.xgb_model_path)
            self.loaded.append("xgb")
            log.info("Loaded XGBoost from %s", settings.xgb_model_path)
        except Exception as e:
            log.warning("XGBoost not loaded: %s", e)

        # CNN
        try:
            self.cnn_model = CharCNN()
            self.cnn_model.load_state_dict(
                torch.load(settings.cnn_model_path, map_location="cpu"))
            self.cnn_model.eval()
            self.loaded.append("cnn")
            log.info("Loaded CharCNN from %s", settings.cnn_model_path)
        except Exception as e:
            log.warning("CharCNN not loaded: %s", e)

        if not self.loaded:
            raise RuntimeError(
                "No models could be loaded. Train one first with "
                "`python -m ml.train_xgb --data data/urls.csv`")

    # ------------------------------------------------------------ predict

    def predict_xgb(self, urls: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not loaded")
        X = pd.DataFrame([extract_features(u) for u in urls])
        proba = self.xgb_model.predict_proba(X)[:, 1]
        return proba, X

    def predict_cnn(self, urls: list[str]) -> np.ndarray:
        if self.cnn_model is None:
            raise RuntimeError("CharCNN model not loaded")
        with torch.no_grad():
            x = torch.tensor([encode_url(u) for u in urls], dtype=torch.long)
            return torch.sigmoid(self.cnn_model(x)).numpy()

    def predict(self, urls: list[str], model: str = "xgb"
                ) -> tuple[np.ndarray, pd.DataFrame | None]:
        if model == "xgb":
            return self.predict_xgb(urls)
        elif model == "cnn":
            return self.predict_cnn(urls), None
        raise ValueError(f"Unknown model: {model}")
