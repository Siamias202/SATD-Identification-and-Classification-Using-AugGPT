"""
=============================================================
SATD Replication  |  Script 2 of 3  —  BiLSTM Identification
=============================================================

Multi-Embedding Version
-----------------------
Supports multiple embedding backbones:

    1. CodeBERT
    2. GraphCodeBERT
    3. CodeT5+
    4. Jina Embeddings v2 Base Code
    5. SFR Embedding Mistral

Pipeline:
    Text
    → Transformer Embedding
    → BiLSTM stack
    → Binary SATD classification

Input:
    binary_train.csv
    binary_val.csv
    binary_test.csv

Output:
    bilstm_best_<embedding>.pt
    <log_name>_<embedding>.txt

Install:
    pip install torch transformers pandas scikit-learn sentence-transformers

Usage Examples:
    python 02_bilstm_identification.py \
        --embedding_model codebert

    python 02_bilstm_identification.py \
        --embedding_model graphcodebert

    python 02_bilstm_identification.py \
        --embedding_model codet5p

    python 02_bilstm_identification.py \
        --embedding_model jina

    python 02_bilstm_identification.py \
        --embedding_model sfr

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
from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    classification_report,
    f1_score
)


# ─────────────────────────────────────────────────────────────
# 1. EMBEDDING CONFIG
# ─────────────────────────────────────────────────────────────

EMBEDDING_MODELS = {

    "codebert": {
        "model_name": "microsoft/codebert-base",
        "dim": 768,
        "type": "hf_cls"
    },

    "graphcodebert": {
        "model_name": "microsoft/graphcodebert-base",
        "dim": 768,
        "type": "hf_cls"
    },

    "codet5p": {
        "model_name": "Salesforce/codet5p-110m-embedding",
        "dim": 256,
        "type": "hf_mean"
    },

    "jina": {
        "model_name": "jinaai/jina-embeddings-v2-base-code",
        "dim": 768,
        "type": "sentence_transformer"
    },

    "sfr": {
        "model_name": "Salesforce/SFR-Embedding-Mistral",
        "dim": 4096,
        "type": "sentence_transformer"
    }
}


# ─────────────────────────────────────────────────────────────
# 2. Universal Embedder
# ─────────────────────────────────────────────────────────────

class UniversalEmbedder:
    """
    Universal embedding wrapper supporting:
        - CodeBERT
        - GraphCodeBERT
        - CodeT5+
        - Jina Embeddings
        - SFR Embedding Mistral
    """

    def __init__(self,
                 embedding_key,
                 device):

        if embedding_key not in EMBEDDING_MODELS:

            raise ValueError(
                f"Unknown embedding model: {embedding_key}"
            )

        self.device = device

        self.cfg = EMBEDDING_MODELS[embedding_key]

        self.embedding_key = embedding_key

        print("\nLoading embedding model:")
        print(f"  Key  : {embedding_key}")
        print(f"  Name : {self.cfg['model_name']}")
        print(f"  Dim  : {self.cfg['dim']}")
        print(f"  Type : {self.cfg['type']}")

        # HuggingFace models
        if self.cfg["type"] in ["hf_cls", "hf_mean"]:

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg["model_name"]
            )

            self.model = AutoModel.from_pretrained(
                self.cfg["model_name"]
            ).to(device)

            self.model.eval()

        # SentenceTransformer models
        elif self.cfg["type"] == "sentence_transformer":

            self.model = SentenceTransformer(
                self.cfg["model_name"],
                device=device
            )

        else:

            raise ValueError(
                f"Unsupported embedding type: "
                f"{self.cfg['type']}"
            )

    @torch.no_grad()
    def embed_batch(self,
                    texts: list,
                    batch_size: int = 32):

        # ─────────────────────────────────────────────
        # HuggingFace Models
        # ─────────────────────────────────────────────

        if self.cfg["type"] in ["hf_cls", "hf_mean"]:

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

                # CLS Pooling
                if self.cfg["type"] == "hf_cls":

                    vecs = out.last_hidden_state[:, 0, :]

                # Mean Pooling
                elif self.cfg["type"] == "hf_mean":

                    token_embeddings = out.last_hidden_state

                    attention_mask = enc["attention_mask"]

                    mask = attention_mask.unsqueeze(-1).expand(
                        token_embeddings.size()
                    ).float()

                    summed = torch.sum(
                        token_embeddings * mask,
                        dim=1
                    )

                    counts = torch.clamp(
                        mask.sum(dim=1),
                        min=1e-9
                    )

                    vecs = summed / counts

                all_vecs.append(
                    vecs.cpu()
                )

            return torch.cat(all_vecs, dim=0)

        # ─────────────────────────────────────────────
        # SentenceTransformer Models
        # ─────────────────────────────────────────────

        elif self.cfg["type"] == "sentence_transformer":

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True
            )

            return embeddings.cpu()


# ─────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):

    def __init__(self,
                 embeddings,
                 labels):

        self.x = embeddings.float()

        self.y = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.x[idx],
            self.y[idx]
        )


# ─────────────────────────────────────────────────────────────
# 4. BiLSTM Model
# ─────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=2,
                 dropout=0.3):

        super().__init__()

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

        self.fc = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # (B, D)

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
# 5. Training
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

        # ───────────────── TRAIN ─────────────────

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

        # ───────────────── VALIDATION ─────────────────

        model.eval()

        val_loss = 0.0

        preds = []

        labels = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)

                yb = yb.to(device)

                out = model(xb)

                val_loss += criterion(
                    out,
                    yb
                ).item()

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

        # Save best model

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
                    f"  ⏹ Early stopping "
                    f"at epoch {epoch}"
                )

                break


# ─────────────────────────────────────────────────────────────
# 6. Evaluation
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
            target_names=[
                "non_debt",
                "SATD"
            ],
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
# 7. Main
# ─────────────────────────────────────────────────────────────

def main(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print(f"BiLSTM SATD | Device: {device}")
    print("=" * 60)

    # ───────────────── LOAD CSV ─────────────────

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

    # ───────────────── EMBEDDINGS ─────────────────

    print("\n[1] Loading embedding model...")

    embedder = UniversalEmbedder(
        embedding_key=args.embedding_model,
        device=device
    )

    embedding_dim = embedder.cfg["dim"]

    print("\n[2] Generating embeddings...")

    train_emb = embedder.embed_batch(
        train_df["text_clean"]
        .astype(str)
        .tolist(),
        batch_size=args.embed_batch_size
    )

    val_emb = embedder.embed_batch(
        val_df["text_clean"]
        .astype(str)
        .tolist(),
        batch_size=args.embed_batch_size
    )

    test_emb = embedder.embed_batch(
        test_df["text_clean"]
        .astype(str)
        .tolist(),
        batch_size=args.embed_batch_size
    )

    print(
        f"  Train embeddings : "
        f"{tuple(train_emb.shape)}"
    )

    print(
        f"  Val embeddings   : "
        f"{tuple(val_emb.shape)}"
    )

    print(
        f"  Test embeddings  : "
        f"{tuple(test_emb.shape)}"
    )

    # ───────────────── DATASETS ─────────────────

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

    # ───────────────── MODEL ─────────────────

    print("\n[4] Initializing model...")

    model = BiLSTMClassifier(
        input_dim=embedding_dim
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters : "
        f"{n_params:,}"
    )

    # ───────────────── TRAIN ─────────────────

    save_path = os.path.join(
        args.processed_dir,
        f"bilstm_best_{args.embedding_model}.pt"
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

    # ───────────────── TEST ─────────────────

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
# 8. CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Embedding "
            "BiLSTM SATD Identification"
        )
    )

    parser.add_argument(
        "--processed_dir",
        default="./processed"
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="codebert",
        choices=[
            "codebert",
            "graphcodebert",
            "codet5p",
            "jina",
            "sfr"
        ],
        help=(
            "Choose embedding model:\n"
            "  codebert      -> microsoft/codebert-base\n"
            "  graphcodebert -> microsoft/graphcodebert-base\n"
            "  codet5p       -> Salesforce/codet5p-110m-embedding\n"
            "  jina          -> jinaai/jina-embeddings-v2-base-code\n"
            "  sfr           -> Salesforce/SFR-Embedding-Mistral"
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=32
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
        default="embedding_training_log"
    )

    args = parser.parse_args()

    os.makedirs(
        args.processed_dir,
        exist_ok=True
    )

    log_path = os.path.join(
        args.processed_dir,
        f"{args.log_name}_{args.embedding_model}.txt"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log_file:

        with redirect_stdout(log_file):

            print("=" * 70)
            print(
                "Multi-Embedding "
                "BiLSTM SATD Training Log"
            )
            print("=" * 70)

            print("\nSelected Arguments:")

            print(
                f"processed_dir   : "
                f"{args.processed_dir}"
            )

            print(
                f"embedding_model : "
                f"{args.embedding_model}"
            )

            print(
                f"batch_size      : "
                f"{args.batch_size}"
            )

            print(
                f"embed_batch_size: "
                f"{args.embed_batch_size}"
            )

            print(
                f"lr              : "
                f"{args.lr}"
            )

            print(
                f"max_epochs      : "
                f"{args.max_epochs}"
            )

            print(
                f"patience        : "
                f"{args.patience}"
            )

            print(
                f"log_name        : "
                f"{args.log_name}"
            )

            print("\n")

            main(args)

    print(
        f"\n✓ Training log saved to: "
        f"{log_path}"
    )