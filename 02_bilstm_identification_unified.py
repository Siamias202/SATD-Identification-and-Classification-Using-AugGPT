"""
=============================================================
SATD Replication  |  Script 2 of 3  —  BiLSTM Identification
=============================================================

CodeBERT Version
----------------
Replaces GloVe embeddings with frozen CodeBERT embeddings.

Pipeline:
    Text
    → CodeBERT [CLS] embedding (768-d)
    → BiLSTM stack
    → Binary SATD classification

Input:
    binary_train.csv
    binary_val.csv
    binary_test.csv

Output:
    bilstm_best.pt
    <log_name>.txt

Install:
    pip install torch transformers pandas scikit-learn

Usage:
    python 02_bilstm_identification.py \
        --processed_dir ./processed \
        --log_name codebert_run

=============================================================
"""

import argparse
import os
from contextlib import redirect_stdout

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    classification_report,
    f1_score
)


# ─────────────────────────────────────────────────────────────
# 1. CodeBERT Embedder
# ─────────────────────────────────────────────────────────────

class CodeBERTEmbedder:
    """
    Produces one 768-d vector per text using
    frozen CodeBERT CLS token embeddings.
    """

    def __init__(self, device):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )

        self.model = AutoModel.from_pretrained(
            "microsoft/codebert-base"
        ).to(device)

        self.model.eval()

    @torch.no_grad()
    def embed_batch(self,
                    texts: list,
                    batch_size: int = 64):

        all_vecs = []

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            out = self.model(**enc)

            # CLS token embedding
            vecs = out.last_hidden_state[:, 0, :].cpu()

            all_vecs.append(vecs)

        return torch.cat(all_vecs, dim=0)


# ─────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):

    def __init__(self,
                 embeddings,
                 labels):

        self.x = embeddings.float()
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.x[idx],
            self.y[idx]
        )


# ─────────────────────────────────────────────────────────────
# 3. BiLSTM Model
# ─────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):

    def __init__(self,
                 input_dim=768,
                 num_classes=2,
                 dropout=0.3):

        super().__init__()

        # reshape input for LSTM
        self.lstm1 = nn.LSTM(
            input_dim,
            128,
            batch_first=True,
            bidirectional=True
        )

        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(256)

        self.lstm2 = nn.LSTM(
            256,
            64,
            batch_first=True,
            bidirectional=True
        )

        self.drop2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(
            128,
            128,
            batch_first=True,
            bidirectional=True
        )

        self.drop3 = nn.Dropout(dropout)

        self.lstm4 = nn.LSTM(
            256,
            128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        # x shape:
        # (B, 768)

        # reshape to sequence length 1
        x = x.unsqueeze(1)

        o1, _ = self.lstm1(x)

        o1 = self.drop1(o1)

        B, T, H = o1.shape

        o1 = self.norm1(
            o1.reshape(B * T, H)
        ).reshape(B, T, H)

        o2, _ = self.lstm2(o1)
        o2 = self.drop2(o2)

        o3, _ = self.lstm3(o2)
        o3 = self.drop3(o3)

        o4, _ = self.lstm4(o3)

        last = o4[:, -1, :]

        return self.fc(last)


# ─────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────

def run_training(model,
                 train_loader,
                 val_loader,
                 device,
                 lr,
                 max_epochs,
                 patience,
                 save_path):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):

        # ───── TRAIN ─────

        model.train()

        train_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            out = model(xb)

            loss = criterion(out, yb)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()

        # ───── VALIDATION ─────

        model.eval()

        val_loss = 0.0

        preds = []
        labels = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                out = model(xb)

                val_loss += criterion(out, yb).item()

                preds.extend(
                    out.argmax(1).cpu().tolist()
                )

                labels.extend(
                    yb.cpu().tolist()
                )

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        macro_f1 = f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={avg_train:.4f} | "
            f"val={avg_val:.4f} | "
            f"val_macro_f1={macro_f1:.4f}"
        )

        # save best model

        if avg_val < best_val_loss:

            best_val_loss = avg_val
            patience_count = 0

            torch.save(
                model.state_dict(),
                save_path
            )

            print("  ✓ checkpoint saved")

        else:

            patience_count += 1

            if patience_count >= patience:

                print(
                    f"  ⏹ Early stopping at epoch {epoch}"
                )

                break


# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────

def run_evaluation(model,
                   loader,
                   device):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            out = model(xb)

            preds.extend(
                out.argmax(1).cpu().tolist()
            )

            labels.extend(
                yb.tolist()
            )

    print("\n── Binary Identification Results ─────────────")

    print(
        classification_report(
            labels,
            preds,
            target_names=["non_debt", "SATD"],
            digits=3,
            zero_division=0
        )
    )

    macro = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    print(f"Macro-avg F1 = {macro:.3f}")


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print(f"CodeBERT + BiLSTM | Device: {device}")
    print("=" * 60)

    # ─────────────────────────────────────────────────
    # LOAD CSV
    # ─────────────────────────────────────────────────

    def load(split):

        path = os.path.join(
            args.processed_dir,
            f"binary_{split}.csv"
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing: {path}"
            )

        return pd.read_csv(path)

    train_df = load("train")
    val_df = load("val")
    test_df = load("test")

    print(
        f"\nRows:"
        f"\n  train = {len(train_df):,}"
        f"\n  val   = {len(val_df):,}"
        f"\n  test  = {len(test_df):,}"
    )

    # ─────────────────────────────────────────────────
    # CODEBERT EMBEDDINGS
    # ─────────────────────────────────────────────────

    print("\n[1] Loading CodeBERT...")

    embedder = CodeBERTEmbedder(device)

    print("\n[2] Generating embeddings...")

    train_emb = embedder.embed_batch(
        train_df["text_clean"].astype(str).tolist(),
        batch_size=args.embed_batch_size
    )

    val_emb = embedder.embed_batch(
        val_df["text_clean"].astype(str).tolist(),
        batch_size=args.embed_batch_size
    )

    test_emb = embedder.embed_batch(
        test_df["text_clean"].astype(str).tolist(),
        batch_size=args.embed_batch_size
    )

    print(f"  Train embeddings : {tuple(train_emb.shape)}")
    print(f"  Val embeddings   : {tuple(val_emb.shape)}")
    print(f"  Test embeddings  : {tuple(test_emb.shape)}")

    # ─────────────────────────────────────────────────
    # DATASETS
    # ─────────────────────────────────────────────────

    print("\n[3] Building dataloaders...")

    train_ds = EmbeddingDataset(
        train_emb,
        train_df["binary_label"].tolist()
    )

    val_ds = EmbeddingDataset(
        val_emb,
        val_df["binary_label"].tolist()
    )

    test_ds = EmbeddingDataset(
        test_emb,
        test_df["binary_label"].tolist()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size
    )

    # ─────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────

    print("\n[4] Initializing model...")

    model = BiLSTMClassifier().to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters : {n_params:,}")

    # ─────────────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────────────

    save_path = os.path.join(
        args.processed_dir,
        "bilstm_best.pt"
    )

    print("\n[5] Training...")

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_path=save_path
    )

    # ─────────────────────────────────────────────────
    # TEST
    # ─────────────────────────────────────────────────

    print("\n[6] Evaluating best model...")

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device
        )
    )

    run_evaluation(
        model,
        test_loader,
        device
    )


# ─────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CodeBERT + BiLSTM SATD Identification"
    )

    parser.add_argument(
        "--processed_dir",
        default="./processed"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default="codebert_training_log"
    )

    args = parser.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)

    log_path = os.path.join(
        args.processed_dir,
        f"{args.log_name}.txt"
    )

    with open(log_path, "w", encoding="utf-8") as log_file:

        with redirect_stdout(log_file):

            print("=" * 70)
            print("CodeBERT + BiLSTM SATD Training Log")
            print("=" * 70)

            print("\nSelected Arguments:")
            print(f"processed_dir   : {args.processed_dir}")
            print(f"batch_size      : {args.batch_size}")
            print(f"embed_batch_size: {args.embed_batch_size}")
            print(f"lr              : {args.lr}")
            print(f"max_epochs      : {args.max_epochs}")
            print(f"patience        : {args.patience}")
            print(f"log_name        : {args.log_name}")

            print("\n")

            main(args)

    print(f"\n✓ Training log saved to: {log_path}")