"""
=============================================================
SATD Replication  |  Script 2 of 3  —  BiLSTM Identification
=============================================================
Paper : Deep Learning and Data Augmentation for Detecting
        Self-Admitted Technical Debt (arXiv:2410.15804)

Input : binary_train/val/test.csv  from 01_preprocessing.py
        (single merged dataset — no artifact separation)

Output: bilstm_best.pt    (saved model weights)

Labels (binary):
    0 = non_debt
    1 = SATD  (code_debt | design_debt | documentation_debt
               | test_debt | requirement_debt)

Architecture (Section III-F):
    Embedding(GloVe-300d)
    → BiLSTM(128) + Dropout(0.3) + BatchNorm
    → BiLSTM(64)  + Dropout(0.3)
    → BiLSTM(128) + Dropout(0.3)
    → BiLSTM(128) [last hidden state]
    → Linear(256 → 2)

GloVe variants:
    glove.6B.50d.txt   —  50d,  69 MB
    glove.6B.100d.txt  — 100d, 171 MB
    glove.6B.200d.txt  — 200d, 342 MB
    glove.6B.300d.txt  — 300d, 462 MB  ← recommended
    glove.42B.300d.txt — 300d, 1.75 GB
    glove.840B.300d.txt— 300d, 2.03 GB

Download:
    wget https://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

Usage:
    pip install torch numpy pandas scikit-learn

    # Recommended (300d)
    python 02_bilstm_identification.py \
        --processed_dir ./processed \
        --glove_path ./glove.6B.300d.txt

    # Different variant — pass matching embed_dim
    python 02_bilstm_identification.py \
        --processed_dir ./processed \
        --glove_path ./glove.6B.200d.txt \
        --embed_dim 200

    # No GloVe (random init)
    python 02_bilstm_identification.py \
        --processed_dir ./processed
=============================================================
"""

import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.UNK: 1}

    def build(self, texts: list, min_freq: int = 1, max_vocab: int = 50_000):
        freq = defaultdict(int)
        for text in texts:
            for tok in str(text).split():
                freq[tok] += 1
        for word, cnt in sorted(freq.items(), key=lambda x: -x[1]):
            if cnt >= min_freq and len(self.word2idx) < max_vocab:
                self.word2idx[word] = len(self.word2idx)
        print(f"  Vocabulary size : {len(self.word2idx):,}")

    def encode(self, text: str, max_len: int) -> list:
        tokens = str(text).split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GloVe loader
# ─────────────────────────────────────────────────────────────────────────────

def load_glove(path: str, vocab: Vocabulary, dim: int) -> np.ndarray:
    """
    Build embedding matrix (vocab_size, dim).
    Words missing from GloVe keep zero vectors. PAD stays zero.
    dim must match the file suffix: 50 | 100 | 200 | 300
    """
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)

    if not path or not os.path.exists(path):
        print("  ⚠  GloVe file not found — using random init (lower performance)")
        rng    = np.random.default_rng(42)
        matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
        matrix[0] = 0.0
        return matrix

    found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab.word2idx:
                matrix[vocab.word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"  GloVe : {found:,} / {len(vocab):,} words matched  (dim={dim})")
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BinaryDataset(Dataset):
    """
    Reads binary_train/val/test.csv
    Required columns: text_clean, binary_label
    """
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int):
        self.x = [vocab.encode(t, max_len) for t in df["text_clean"].astype(str)]
        self.y = df["binary_label"].astype(int).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BiLSTM Model  (Section III-F)
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Stacked BiLSTM for binary classification (non_debt vs SATD).

        Embedding  (vocab_size, embed_dim)
        BiLSTM-1   hidden=128 → out=256  Dropout(0.3)  BatchNorm1d(256)
        BiLSTM-2   hidden= 64 → out=128  Dropout(0.3)
        BiLSTM-3   hidden=128 → out=256  Dropout(0.3)
        BiLSTM-4   hidden=128 → out=256  [last time-step only]
        Linear(256 → num_classes)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 embed_matrix: np.ndarray,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32)
        )

        self.lstm1 = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(256)

        self.lstm2 = nn.LSTM(256, 64,  batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop3 = nn.Dropout(dropout)

        self.lstm4 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        e = self.embedding(x)                       # (B, T, E)

        o1, _ = self.lstm1(e)                       # (B, T, 256)
        o1     = self.drop1(o1)
        B, T, H = o1.shape
        o1     = self.norm1(o1.reshape(B * T, H)).reshape(B, T, H)

        o2, _ = self.lstm2(o1)                      # (B, T, 128)
        o2     = self.drop2(o2)

        o3, _ = self.lstm3(o2)                      # (B, T, 256)
        o3     = self.drop3(o3)

        o4, _ = self.lstm4(o3)                      # (B, T, 256)
        last   = o4[:, -1, :]                       # (B, 256)

        return self.fc(last)                        # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, device,
                 lr, max_epochs, patience, save_path):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):

        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # validate
        model.eval()
        val_loss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out     = model(xb)
                val_loss += criterion(out, yb).item()
                preds.extend(out.argmax(1).cpu().tolist())
                labels.extend(yb.cpu().tolist())

        avg_val  = val_loss / len(val_loader)
        avg_tr   = train_loss / len(train_loader)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

        print(f"  Epoch {epoch:03d} | "
              f"train={avg_tr:.4f}  val={avg_val:.4f}  "
              f"val_macro_f1={macro_f1:.4f}")

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(yb.tolist())

    print("\n── Binary Identification Results ───────────────────────────────")
    print(classification_report(
        labels, preds,
        target_names=["non_debt", "SATD"],
        digits=3, zero_division=0,
    ))
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"  Macro-avg F1 = {macro:.3f}")
    return preds, labels


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BiLSTM  |  Binary Identification  |  Device: {device}")
    print(f"{'='*60}\n")

    def load(split):
        path = os.path.join(args.processed_dir, f"binary_{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\nRun 01_preprocessing.py first."
            )
        return pd.read_csv(path)

    train_df = load("train")
    val_df   = load("val")
    test_df  = load("test")

    print(f"  Rows → train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")
    print(f"\n  Label distribution (train):")
    print("  ", train_df["binary_label"]
                 .map({0: "non_debt", 1: "SATD"})
                 .value_counts().to_string())

    # Vocabulary
    print("\n[1] Building vocabulary from training set...")
    vocab = Vocabulary()
    vocab.build(train_df["text_clean"].tolist())

    # GloVe
    print(f"\n[2] Loading GloVe  (dim={args.embed_dim})...")
    embed_matrix = load_glove(args.glove_path, vocab, args.embed_dim)

    # Loaders
    print("\n[3] Building data loaders...")
    def make_loader(df, shuffle=False):
        ds = BinaryDataset(df, vocab, args.max_len)
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    # Model
    print("\n[4] Initialising BiLSTM model...")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        embed_matrix=embed_matrix,
        num_classes=2,
        dropout=0.3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    # Train
    save_path = os.path.join(args.processed_dir, "bilstm_best.pt")
    print(f"\n[5] Training  (max_epochs={args.max_epochs}, patience={args.patience})...")
    run_training(
        model, train_loader, val_loader, device,
        lr=args.lr, max_epochs=args.max_epochs,
        patience=args.patience, save_path=save_path,
    )

    # Evaluate
    print(f"\n[6] Loading best checkpoint and evaluating on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    run_evaluation(model, test_loader, device)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BiLSTM Binary Identification — Script 2/3"
    )
    parser.add_argument(
        "--processed_dir", default="./processed",
        help="Folder with binary_train/val/test.csv from Script 1"
    )
    parser.add_argument(
        "--glove_path", default="",
        help="GloVe .txt file (recommended: glove.6B.300d.txt). "
             "Leave blank for random init."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=300,
        help="Must match GloVe file: 50 | 100 | 200 | 300  (default: 300)"
    )
    parser.add_argument("--max_len",    type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int,   default=50)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()
    main(args)