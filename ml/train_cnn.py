"""
ml/train_cnn.py
Character-level CNN baseline. Tokenizes raw URLs at the character level and
learns its own representation, no engineered features.

Architecture (small but effective):
    Embedding(vocab=128, dim=64)
    Conv1d(64 -> 128, k=3) + ReLU + MaxPool
    Conv1d(128 -> 128, k=3) + ReLU + MaxPool
    Conv1d(128 -> 128, k=3) + ReLU + GlobalMaxPool
    Linear(128 -> 64) + Dropout(0.3) + ReLU
    Linear(64 -> 1)

Usage:
    python -m ml.train_cnn --data data/urls.csv --out models/cnn/v1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, classification_report)
from torch.utils.data import DataLoader, Dataset

from ml.data_utils import load_dataset, split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_cnn")

MAX_LEN = 200
VOCAB_SIZE = 128       # printable ASCII covers most URLs
PAD_IDX = 0


def encode_url(url: str, max_len: int = MAX_LEN) -> list[int]:
    """Map each char to its ASCII code, truncate or pad to max_len."""
    ids = [min(ord(c), VOCAB_SIZE - 1) for c in url[:max_len]]
    return ids + [PAD_IDX] * (max_len - len(ids))


class URLDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int]):
        self.x = torch.tensor([encode_url(u) for u in urls], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class CharCNN(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, emb_dim=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq)
        e = self.embed(x).transpose(1, 2)        # (batch, emb, seq)
        h = torch.relu(self.conv1(e))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = self.pool(h)
        h = torch.relu(self.conv3(h))
        h = torch.amax(h, dim=2)                  # global max pool
        h = self.dropout(torch.relu(self.fc1(h)))
        return self.fc2(h).squeeze(-1)            # raw logits


def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p.extend(probs.tolist())
            all_y.extend(y.numpy().tolist())
    return np.array(all_y), np.array(all_p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models/cnn/v1")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    df = load_dataset(args.data)
    train_df, val_df, test_df = split(df, seed=args.seed)

    train_loader = DataLoader(
        URLDataset(train_df["url"].tolist(), train_df["label"].tolist()),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        URLDataset(val_df["url"].tolist(), val_df["label"].tolist()),
        batch_size=args.batch_size)
    test_loader = DataLoader(
        URLDataset(test_df["url"].tolist(), test_df["label"].tolist()),
        batch_size=args.batch_size)

    model = CharCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        y_val, p_val = evaluate(model, val_loader, device)
        auc = roc_auc_score(y_val, p_val)
        log.info(
            "epoch %d  train_loss=%.4f  val_auc=%.4f",
            epoch, running_loss / len(train_loader.dataset), auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), out_dir / "model.pt")
            log.info("  saved new best (AUC=%.4f)", auc)

    # Final test eval with best weights
    model.load_state_dict(torch.load(out_dir / "model.pt"))
    y_test, p_test = evaluate(model, test_loader, device)
    pred = (p_test >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "best_val_auc": float(best_auc),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "max_len": MAX_LEN,
        "vocab_size": VOCAB_SIZE,
    }
    log.info("Test: %s", json.dumps(metrics, indent=2))
    log.info("\n%s", classification_report(
        y_test, pred, target_names=["benign", "phish"]))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    card = {
        "model_name": "phishing-url-charcnn",
        "version": out_dir.name,
        "framework": f"pytorch=={torch.__version__}",
        "architecture": "Embedding(128,64) -> 3x Conv1d(128) -> GMaxPool -> FC(64) -> FC(1)",
        "task": "binary classification (phish vs benign URL)",
        "metrics": metrics,
        "intended_use": "Comparison baseline against engineered-feature XGBoost",
        "limitations": (
            "No engineered features means worse interpretability. "
            "Marginal accuracy gain over XGBoost rarely justifies the "
            "deployment overhead unless URL strings are very long."),
    }
    with open(out_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)


if __name__ == "__main__":
    main()
