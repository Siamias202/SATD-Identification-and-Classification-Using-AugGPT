"""
=============================================================
SATD Replication  |  Script 3 of 3  —  Categorization
          (Multi-Model: BERT / RoBERTa / CodeBERT / DeBERTa-v3)
=============================================================
Paper : Deep Learning and Data Augmentation for Detecting
        Self-Admitted Technical Debt (arXiv:2410.15804)

Input : category_train/val/test.csv  from 01_preprocessing.py
        (SATD-only rows from the single merged dataset)

Output: {model}_best.pt    (saved model weights)

Labels (6-class):
    0 = non_debt            ← kept so the head matches category_label
    1 = code_debt
    2 = documentation_debt
    3 = test_debt
    4 = requirement_debt
    5 = design_debt

    NOTE: category_train/val/test.csv contains SATD rows only
          (binary_label == 1) so non_debt rows will not appear
          in practice. The head dimension is kept at 6 for
          label-index consistency with 01_preprocessing.py.

Models (ranked by expected SATD performance):
    deberta   microsoft/deberta-v3-base   ← best overall
    codebert  microsoft/codebert-base     ← best for code/commit text
    roberta   roberta-base                ← strong general baseline
    bert      bert-base-uncased           ← paper baseline

Usage:
    pip install torch transformers pandas scikit-learn

    # Recommended
    python 03_categorization.py --model deberta

    # CodeBERT (great for code_debt and design_debt)
    python 03_categorization.py --model codebert

    # RoBERTa
    python 03_categorization.py --model roberta

    # Paper baseline
    python 03_categorization.py --model bert

    # Run all 4 models and print comparison table
    python 03_categorization.py --run_all

    # Predict a single text (requires saved .pt checkpoint)
    python 03_categorization.py --model deberta \
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


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Must match CATEGORY_MAP in 01_preprocessing.py exactly
ID2LABEL = {
    0: "non_debt",
    1: "code_debt",
    2: "documentation_debt",
    3: "test_debt",
    4: "requirement_debt",
    5: "design_debt",
}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = len(ID2LABEL)     # 6

# HuggingFace checkpoint IDs
MODEL_MAP = {
    "bert":     "bert-base-uncased",
    "roberta":  "roberta-base",
    "codebert": "microsoft/codebert-base",
    "deberta":  "microsoft/deberta-v3-base",
}

# Model-specific recommended learning rates
LR_MAP = {
    "bert":     5e-5,
    "roberta":  2e-5,
    "codebert": 2e-5,
    "deberta":  1e-5,   # DeBERTa-v3 benefits from a lower LR
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CategoryDataset(Dataset):
    """
    Tokenises text_clean with any HuggingFace AutoTokenizer.
    Required columns: text_clean, category_label
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Classifier  (shared architecture for all 4 models)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerCategorizer(nn.Module):
    """
    Any HuggingFace encoder backbone
    → [CLS] pooled output  (768-d for all four models)
    → Linear(768 → 256) + ReLU + Dropout
    → Linear(256 → 6)

    DeBERTa-v3 does not expose pooler_output by default;
    we fall back to last_hidden_state[:, 0, :] automatically.
    """

    def __init__(self, model_name: str, hidden_dim: int = 256,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        # Force encoder to float32 — DeBERTa-v3 can load as float16
        # in some environments (Colab, certain CUDA builds), which causes
        # a dtype mismatch with the float32 Linear layers below.
        self.encoder    = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model_name = model_name
        enc_dim         = self.encoder.config.hidden_size   # 768

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(enc_dim, hidden_dim)       # float32
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_dim, num_classes)   # float32

    def _cls(self, outputs):
        """
        Return [CLS] vector regardless of model family,
        always cast to float32 to prevent Half/Float dtype errors.
        """
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.float()            # BERT / RoBERTa / CodeBERT
        return outputs.last_hidden_state[:, 0, :].float()  # DeBERTa-v3

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._cls(out)                     # (B, 768) — always float32
        x = self.dropout(pooled)
        x = self.relu(self.fc1(x))                 # (B, 256)
        return self.fc2(x)                          # (B, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, device,
                 lr, max_epochs, patience, save_path):
    """
    AdamW + linear warmup + CrossEntropyLoss + early stopping on val_loss.
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

        # train
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

        # validate
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
            print(f"    ✓ checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch}")
                break


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, loader, device, verbose: bool = True):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)
            preds.extend(out.argmax(1).cpu().tolist())
            gold.extend(batch["label"].tolist())

    # Only report classes that actually appear in the test set
    present = sorted(set(gold))
    target_names = [ID2LABEL[i] for i in present]

    macro    = f1_score(gold, preds, average="macro", zero_division=0)
    per_cls  = f1_score(gold, preds, average=None,
                        labels=present, zero_division=0)

    if verbose:
        print("\n── Categorization Results ───────────────────────────────────")
        print(classification_report(
            gold, preds,
            labels=present,
            target_names=target_names,
            digits=3, zero_division=0,
        ))
        print(f"  Macro-avg F1 = {macro:.3f}")

    return macro, dict(zip(target_names, per_cls.tolist())), preds, gold


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Single-text prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_one(text: str, model, tokenizer, device, max_len: int) -> str:
    model.eval()
    enc = tokenizer(
        text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return ID2LABEL[logits.argmax(1).item()]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Single-model pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(model_key: str, processed_dir: str,
                 max_len: int, batch_size: int, lr: float,
                 max_epochs: int, patience: int,
                 device: torch.device,
                 predict_text: str = "") -> dict:

    hf_id     = MODEL_MAP[model_key]
    save_path = os.path.join(processed_dir, f"{model_key}_best.pt")

    print(f"\n{'='*62}")
    print(f"  Model  : {model_key.upper()}  ({hf_id})")
    print(f"  Device : {device}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"{'='*62}")

    print("\n[1] Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    # ── Predict-only mode ─────────────────────────────────────────────────────
    if predict_text:
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"No checkpoint at {save_path}. Run training first."
            )
        print(f"  Loading weights from {save_path}...")
        model = TransformerCategorizer(hf_id).to(device).float()
        model.load_state_dict(torch.load(save_path, map_location=device))
        label = predict_one(predict_text, model, tokenizer, device, max_len)
        print(f"\n  Input               : {predict_text!r}")
        print(f"  Predicted SATD type : {label}")
        return {}

    # ── Load splits ───────────────────────────────────────────────────────────
    def load(split):
        path = os.path.join(processed_dir, f"category_{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\nRun 01_preprocessing.py first."
            )
        return pd.read_csv(path)

    train_df = load("train")
    val_df   = load("val")
    test_df  = load("test")

    print(f"\n  Rows → train:{len(train_df):,}  "
          f"val:{len(val_df):,}  test:{len(test_df):,}")
    print("\n  Class distribution (train):")
    print("  ", train_df["category_label"]
                 .map(ID2LABEL).value_counts().to_string())

    # ── Loaders ───────────────────────────────────────────────────────────────
    print("\n[2] Building data loaders...")
    def make_loader(df, shuffle=False):
        ds = CategoryDataset(df, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[3] Initialising {model_key.upper()} classifier...")
    model = TransformerCategorizer(hf_id, num_classes=NUM_CLASSES).to(device)
    model = model.float()   # guarantee all params are float32 after device move
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[4] Training  (lr={lr}, patience={patience})...")
    run_training(
        model, train_loader, val_loader, device,
        lr=lr, max_epochs=max_epochs,
        patience=patience, save_path=save_path,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[5] Evaluating best checkpoint...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    macro, per_cls, _, _ = run_evaluation(model, test_loader, device)

    return {"model": model_key, "macro_f1": macro, **per_cls}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Compare all models
# ─────────────────────────────────────────────────────────────────────────────

def run_all(args, device):
    """Train & evaluate every model, then print a ranked comparison table."""
    results = []
    for model_key in MODEL_MAP:
        lr = args.lr if args.lr else LR_MAP[model_key]
        try:
            r = run_pipeline(
                model_key=model_key,
                processed_dir=args.processed_dir,
                max_len=args.max_len,
                batch_size=args.batch_size,
                lr=lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
            )
            if r:
                results.append(r)
        except FileNotFoundError as e:
            print(f"  ⚠  Skipping {model_key}: {e}")

    if not results:
        return

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    print("\n\n" + "="*70)
    print("  COMPARISON TABLE  —  sorted by Macro-avg F1  (↑ higher is better)")
    print("="*70)
    print(df.round(3).to_string(index=False))

    out_path = os.path.join(args.processed_dir, "comparison_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_all:
        run_all(args, device)
        return

    lr = args.lr if args.lr else LR_MAP[args.model]
    run_pipeline(
        model_key=args.model,
        processed_dir=args.processed_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        predict_text=args.predict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SATD Categorization — Script 3/3 (Multi-Model)"
    )
    parser.add_argument(
        "--model", default="deberta",
        choices=list(MODEL_MAP.keys()),
        help="bert | roberta | codebert | deberta  (default: deberta)"
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help="Train all 4 models and print a comparison table"
    )
    parser.add_argument(
        "--processed_dir", default="./processed",
        help="Folder with category_train/val/test.csv from Script 1"
    )
    parser.add_argument("--max_len",    type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument(
        "--lr", type=float, default=0,
        help="Learning rate. 0 = use model-specific default "
             "(deberta:1e-5, roberta/codebert:2e-5, bert:5e-5)"
    )
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience",   type=int, default=3)
    parser.add_argument(
        "--predict", default="",
        help="Predict SATD type for this text string "
             "(requires a saved .pt checkpoint)"
    )
    args = parser.parse_args()
    main(args)