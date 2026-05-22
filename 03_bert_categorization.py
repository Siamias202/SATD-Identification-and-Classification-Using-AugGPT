"""
=============================================================
SATD Replication  |  Script 3 of 3  —  Categorization
                     (Multi-Model: BERT / RoBERTa / CodeBERT / DeBERTa-v3)
=============================================================
Paper : Deep Learning and Data Augmentation for Detecting
        Self-Admitted Technical Debt (arXiv:2410.15804)

This script replaces the original bert-only categorizer with
four selectable models ranked by expected performance on SATD:

  Rank  Model              HuggingFace ID                    Why better
  ----  -----------------  --------------------------------  -------------------------------
  1 ★  DeBERTa-v3-base    microsoft/deberta-v3-base         Disentangled attention + RTD;
                                                             best on NLP benchmarks (GLUE/SQuAD)
  2     CodeBERT-base      microsoft/codebert-base           Pre-trained on code + NL pairs;
                                                             best for CC/CM artifacts
  3     RoBERTa-base       roberta-base                      BERT trained longer, dynamic
                                                             masking, no NSP; consistently
                                                             beats BERT on classification
  4     BERT-base-uncased  bert-base-uncased                 Paper baseline (included for
                                                             direct comparison)

Usage:
    pip install torch transformers pandas scikit-learn

    # DeBERTa-v3 (recommended — best overall)
    python 03_categorization.py --model deberta --artifact CC

    # CodeBERT (best for code comments & commit messages)
    python 03_categorization.py --model codebert --artifact CC

    # RoBERTa
    python 03_categorization.py --model roberta --artifact CC

    # BERT (paper baseline)
    python 03_categorization.py --model bert --artifact CC

    # Run all 4 models on all 4 artifacts and compare
    python 03_categorization.py --run_all --processed_dir ./processed

    # Predict a single text after training
    python 03_categorization.py --model deberta --artifact CC \
        --predict "TODO: this is a temporary hack, fix before release"
=============================================================
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

# ── Constants ─────────────────────────────────────────────────────────────────

SATD_TYPES = {"code_debt", "design_debt", "requirement_debt", "documentation_debt","test_debt"}
LABEL2ID  = {v: k for k, v in SATD_TYPES.items()}
NUM_CLASSES = 5
ARTIFACTS   = ["CC", "IS", "PS", "CM"]

# HuggingFace model IDs
MODEL_MAP = {
    "bert":     "bert-base-uncased",
    "roberta":  "roberta-base",
    "codebert": "microsoft/codebert-base",
    "deberta":  "microsoft/deberta-v3-base",
}

# Recommended learning rates per model family
LR_MAP = {
    "bert":     5e-5,
    "roberta":  2e-5,
    "codebert": 2e-5,
    "deberta":  1e-5,   # DeBERTa benefits from a lower LR
}


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class CategoryDataset(Dataset):
    """
    Tokenises text with whichever AutoTokenizer the chosen model uses.
    All four models accept the same interface (input_ids + attention_mask).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
        self.texts   = df["text_clean"].astype(str).tolist()
        self.labels  = df["category_label"].astype(int).tolist()
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── 2. Classifier head (shared by all models) ─────────────────────────────────

class TransformerCategorizer(nn.Module):
    """
    Any HuggingFace encoder  →  [CLS] pooled output
    → Linear(hidden → 256) + ReLU + Dropout
    → Linear(256 → 4)

    Works with BERT, RoBERTa, CodeBERT, and DeBERTa-v3.
    DeBERTa-v3 uses 'last_hidden_state[:,0,:]' because it does not
    expose a pooler_output by default — handled automatically below.
    """

    def __init__(self, model_name: str, hidden_dim: int = 256,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        enc_hidden      = self.encoder.config.hidden_size   # 768 for all four

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(enc_hidden, hidden_dim)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_dim, num_classes)

    def _pool(self, outputs):
        """Return [CLS] vector regardless of model family."""
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output          # BERT, RoBERTa, CodeBERT
        return outputs.last_hidden_state[:, 0, :] # DeBERTa-v3

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self._pool(outputs)              # (B, 768)
        x = self.dropout(pooled)
        x = self.relu(self.fc1(x))                # (B, 256)
        return self.fc2(x)                         # (B, 4)


# ── 3. Training ───────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device,
          lr, max_epochs, patience, save_path):
    """
    Fine-tuning with:
      AdamW + linear warmup (10 % of steps)
      CrossEntropyLoss
      Gradient clipping (max norm = 1.0)
      Early stopping on val_loss
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01
    )
    total_steps = len(train_loader) * max_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ── validate ──
        model.eval()
        val_loss, preds, gold = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl  = batch["label"].to(device)
                out  = model(ids, mask)
                val_loss += criterion(out, lbl).item()
                preds.extend(out.argmax(1).cpu().tolist())
                gold.extend(lbl.cpu().tolist())

        avg_val  = val_loss / len(val_loader)
        avg_tr   = train_loss / len(train_loader)
        macro_f1 = f1_score(gold, preds, average="macro", zero_division=0)
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch:02d} | "
              f"train={avg_tr:.4f}  val={avg_val:.4f}  "
              f"val_macro_f1={macro_f1:.4f}  ({elapsed:.1f}s)")

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch}")
                break

    return best_val_loss


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model, loader, device, verbose: bool = True):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)
            preds.extend(out.argmax(1).cpu().tolist())
            gold.extend(batch["label"].tolist())

    macro = f1_score(gold, preds, average="macro", zero_division=0)
    per_class = f1_score(gold, preds, average=None,
                         labels=[0,1,2,3], zero_division=0)

    if verbose:
        print("\n── Categorization Results ───────────────────────────────────────")
        print(classification_report(
            gold, preds,
            target_names=[ID2LABEL[i] for i in range(4)],
            digits=3, zero_division=0,
        ))
        print(f"  Macro-avg F1 = {macro:.3f}")

    return macro, per_class, preds, gold


# ── 5. Single inference ───────────────────────────────────────────────────────

def predict_one(text: str, model, tokenizer, device, max_len: int = 128) -> str:
    model.eval()
    enc = tokenizer(text, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device),
                       enc["attention_mask"].to(device))
    return ID2LABEL[logits.argmax(1).item()]


# ── 6. Single-model pipeline ──────────────────────────────────────────────────

def run_pipeline(model_key: str, artifact: str, processed_dir: str,
                 max_len: int, batch_size: int, lr: float,
                 max_epochs: int, patience: int, device: torch.device,
                 predict_text: str = "") -> dict:
    """
    Full train → evaluate pipeline for one (model, artifact) pair.
    Returns dict with per-class and macro F1.
    """
    hf_id     = MODEL_MAP[model_key]
    save_path = f"{model_key}_{artifact}_best.pt"

    print(f"\n{'='*62}")
    print(f"  Model    : {model_key.upper()}  ({hf_id})")
    print(f"  Artifact : {artifact}")
    print(f"  Device   : {device}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"{'='*62}")

    # Tokeniser
    print("\n[1] Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # ── Predict-only mode ─────────────────────────────────────────────────────
    if predict_text:
        print(f"[predict] Loading weights from {save_path}")
        model = TransformerCategorizer(hf_id).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        label = predict_one(predict_text, model, tokenizer, device, max_len)
        print(f"\n  Input : {predict_text!r}")
        print(f"  SATD type predicted : {label}")
        return {}

    # Load splits
    def load(split):
        p = os.path.join(processed_dir, f"{artifact}_category_{split}.csv")
        return pd.read_csv(p)

    train_df, val_df, test_df = load("train"), load("val"), load("test")
    print(f"\n  Rows → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    print("  Class dist (train):")
    print("  ", train_df["category_label"].map(ID2LABEL).value_counts().to_string())

    # Data loaders
    print("\n[2] Building data loaders...")
    def make_loader(df, shuffle=False):
        ds = CategoryDataset(df, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    # Model
    print(f"\n[3] Initialising {model_key.upper()} classifier...")
    model = TransformerCategorizer(hf_id).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Train
    print(f"\n[4] Training (patience={patience}, max_epochs={max_epochs})...")
    train(model, train_loader, val_loader, device,
          lr=lr, max_epochs=max_epochs,
          patience=patience, save_path=save_path)

    # Evaluate
    print(f"\n[5] Evaluating best checkpoint ({save_path})...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    macro, per_class, _, _ = evaluate(model, test_loader, device)

    return {
        "model":    model_key,
        "artifact": artifact,
        "macro_f1": macro,
        "C/D_f1":   per_class[0],
        "DOC_f1":   per_class[1],
        "TES_f1":   per_class[2],
        "REQ_f1":   per_class[3],
    }


# ── 7. Compare all models × all artifacts ────────────────────────────────────

def run_all(args, device):
    """
    Train and evaluate every (model, artifact) combination and print a
    comparison table so you can see which model wins per artifact.
    """
    results = []
    for model_key in MODEL_MAP:
        lr = args.lr if args.lr else LR_MAP[model_key]
        for artifact in ARTIFACTS:
            try:
                r = run_pipeline(
                    model_key=model_key,
                    artifact=artifact,
                    processed_dir=args.processed_dir,
                    max_len=args.max_len,
                    batch_size=args.batch_size,
                    lr=lr,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    device=device,
                )
                results.append(r)
            except FileNotFoundError as e:
                print(f"  ⚠️  Skipping {model_key}/{artifact}: {e}")

    if not results:
        return

    df = pd.DataFrame(results)
    df = df.sort_values(["artifact", "macro_f1"], ascending=[True, False])

    print("\n\n" + "="*70)
    print("  COMPARISON TABLE  —  Macro-avg F1  (higher is better)")
    print("="*70)
    pivot = df.pivot(index="model", columns="artifact", values="macro_f1")
    print(pivot.round(3).to_string())

    print("\n  Full per-class breakdown:")
    print(df[["model","artifact","C/D_f1","DOC_f1","TES_f1","REQ_f1","macro_f1"]]
          .round(3).to_string(index=False))

    csv_path = "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_all:
        run_all(args, device)
        return

    lr = args.lr if args.lr else LR_MAP[args.model]
    run_pipeline(
        model_key=args.model,
        artifact=args.artifact,
        processed_dir=args.processed_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        predict_text=args.predict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SATD Categorization — Script 3/3 (Multi-Model)"
    )
    # Model & artifact
    parser.add_argument("--model", default="deberta",
                        choices=list(MODEL_MAP.keys()),
                        help="Model to use: bert | roberta | codebert | deberta")
    parser.add_argument("--artifact", default="CC",
                        choices=ARTIFACTS,
                        help="Artifact to train on: CC | IS | PS | CM")
    parser.add_argument("--run_all", action="store_true",
                        help="Train all 4 models × 4 artifacts and print comparison")

    # Paths
    parser.add_argument("--processed_dir", default="./processed",
                        help="Folder with *_category_*.csv from Script 1")

    # Hyperparameters
    parser.add_argument("--max_len",    type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0,
                        help="Learning rate. 0 = use model-specific default "
                             "(deberta:1e-5, roberta/codebert:2e-5, bert:5e-5)")
    parser.add_argument("--max_epochs", type=int,   default=10)
    parser.add_argument("--patience",   type=int,   default=3,
                        help="Early stopping patience on val loss")

    # Inference
    parser.add_argument("--predict", default="",
                        help="If set, predict the SATD type of this text string "
                             "(requires a saved .pt checkpoint)")

    args = parser.parse_args()
    main(args)