"""
ml/evaluate.py
Side-by-side evaluation of XGBoost vs CharCNN on the same test set.
Outputs a markdown table you can drop straight into the README.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              roc_auc_score)

from ml.data_utils import load_dataset, split
from ml.features import extract_features
from ml.train_cnn import CharCNN, encode_url, MAX_LEN


def time_inference(predict_fn, items, n_warmup: int = 50):
    for x in items[:n_warmup]:
        predict_fn([x])
    t0 = time.perf_counter()
    for x in items:
        predict_fn([x])
    elapsed = time.perf_counter() - t0
    return (elapsed / len(items)) * 1000  # ms per URL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--xgb", default="models/xgb/v1")
    p.add_argument("--cnn", default="models/cnn/v1")
    p.add_argument("--out", default="docs/comparison.md")
    args = p.parse_args()

    df = load_dataset(args.data)
    _, _, test_df = split(df)
    urls = test_df["url"].tolist()
    y = test_df["label"].values

    # ---- XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{args.xgb}/model.json")
    X = pd.DataFrame([extract_features(u) for u in urls])
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    xgb_latency = time_inference(
        lambda batch: xgb_model.predict_proba(
            pd.DataFrame([extract_features(u) for u in batch])),
        urls[:200])

    # ---- CharCNN
    cnn = CharCNN()
    cnn.load_state_dict(torch.load(f"{args.cnn}/model.pt", map_location="cpu"))
    cnn.eval()
    with torch.no_grad():
        x = torch.tensor([encode_url(u) for u in urls], dtype=torch.long)
        cnn_proba = torch.sigmoid(cnn(x)).numpy()
    cnn_pred = (cnn_proba >= 0.5).astype(int)

    def cnn_predict(batch):
        with torch.no_grad():
            xb = torch.tensor([encode_url(u) for u in batch], dtype=torch.long)
            return torch.sigmoid(cnn(xb)).numpy()
    cnn_latency = time_inference(cnn_predict, urls[:200])

    rows = []
    for name, pred, proba, lat in [
        ("XGBoost (engineered features)", xgb_pred, xgb_proba, xgb_latency),
        ("CharCNN (raw URL)", cnn_pred, cnn_proba, cnn_latency),
    ]:
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, proba),
            "latency_ms": lat,
        })
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Model comparison\n\n")
        f.write(f"Test set size: {len(urls)}\n\n")
        f.write(cmp.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(f"\nWrote comparison to {out}")


if __name__ == "__main__":
    main()
