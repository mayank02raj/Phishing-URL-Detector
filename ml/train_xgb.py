"""
ml/train_xgb.py
Train an XGBoost classifier on engineered URL features.

Usage:
    python -m ml.train_xgb --data data/urls.csv --out models/xgb/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score)

from ml.data_utils import load_dataset, split
from ml.features import extract_features, feature_names

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_xgb")


def vectorize(urls: list[str]) -> pd.DataFrame:
    return pd.DataFrame([extract_features(u) for u in urls])


def find_best_threshold(y_true, y_proba, target_precision: float = 0.95):
    """Pick the threshold that maximizes recall subject to precision >= target."""
    p, r, t = precision_recall_curve(y_true, y_proba)
    valid = p >= target_precision
    if not valid.any():
        return 0.5
    idx = np.argmax(r[valid])
    valid_idx = np.where(valid)[0][idx]
    if valid_idx >= len(t):
        return 0.5
    return float(t[valid_idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models/xgb/v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-precision", type=float, default=0.95)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset")
    df = load_dataset(args.data)
    train_df, val_df, test_df = split(df, seed=args.seed)
    log.info("Split: train=%d val=%d test=%d",
             len(train_df), len(val_df), len(test_df))

    log.info("Extracting features")
    X_train = vectorize(train_df["url"].tolist())
    X_val = vectorize(val_df["url"].tolist())
    X_test = vectorize(test_df["url"].tolist())
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    log.info("Training XGBoost")
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Pick threshold on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_proba, args.target_precision)
    log.info("Selected threshold=%.4f (target precision >= %.2f)",
             threshold, args.target_precision)

    # Test set evaluation
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "threshold": threshold,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_features": X_train.shape[1],
        "best_iteration": int(model.best_iteration or model.n_estimators),
    }

    log.info("Test metrics: %s", json.dumps(metrics, indent=2))
    log.info("Confusion matrix:\n%s", confusion_matrix(y_test, test_pred))
    log.info("\n%s", classification_report(
        y_test, test_pred, target_names=["benign", "phish"]))

    # Feature importance
    importance = sorted(
        zip(feature_names(), model.feature_importances_),
        key=lambda x: x[1], reverse=True)
    log.info("Top 15 features:")
    for name, score in importance[:15]:
        log.info("  %-22s %.4f", name, score)

    # Persist artifacts
    model.save_model(out_dir / "model.json")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names(), f, indent=2)

    # Save reference feature distributions for drift monitoring
    ref_stats = {
        "feature_means": X_train.mean().to_dict(),
        "feature_stds": X_train.std().to_dict(),
        "feature_quantiles": {
            col: X_train[col].quantile([0.1, 0.5, 0.9]).tolist()
            for col in X_train.columns
        },
    }
    with open(out_dir / "reference_stats.json", "w") as f:
        json.dump(ref_stats, f, indent=2)

    # Model card
    card = {
        "model_name": "phishing-url-xgb",
        "version": out_dir.name,
        "framework": f"xgboost=={xgb.__version__}",
        "task": "binary classification (phish vs benign URL)",
        "training_data": args.data,
        "n_features": len(feature_names()),
        "features": feature_names(),
        "metrics": metrics,
        "intended_use": (
            "Email gateway, browser extension, SOC URL triage. "
            "Not a substitute for sandboxing or threat intel lookups."),
        "limitations": (
            "Trained on URL strings only, no DOM or WHOIS context. "
            "Performance degrades on adversarial homoglyphs and on TLDs "
            "underrepresented in training data."),
    }
    with open(out_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    log.info("Saved artifacts to %s", out_dir)


if __name__ == "__main__":
    main()
