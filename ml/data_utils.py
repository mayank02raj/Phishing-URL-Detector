"""
ml/data_utils.py
Dataset loading, splitting, and basic sanity checks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV with columns url,label. Robust to a few common variants."""
    df = pd.read_csv(path)
    # Common alternate column names
    rename = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("url", "urls"):
            rename[col] = "url"
        elif cl in ("label", "class", "result", "is_phish"):
            rename[col] = "label"
    df = df.rename(columns=rename)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must have url and label columns, got {list(df.columns)}")

    df = df[["url", "label"]].dropna()
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 4]

    # Normalize labels: phish=1, benign=0
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({
            "phishing": 1, "phish": 1, "bad": 1, "1": 1, "true": 1,
            "benign": 0, "legitimate": 0, "good": 0, "0": 0, "false": 0,
        })
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    log.info(
        "Loaded %d rows (%d phish / %d benign)",
        len(df), int(df["label"].sum()),
        int((1 - df["label"]).sum()))
    return df


def split(df: pd.DataFrame, test_size: float = 0.15,
          val_size: float = 0.15, seed: int = 42):
    """Stratified train/val/test split."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train, test_size=relative_val,
        stratify=train["label"], random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
