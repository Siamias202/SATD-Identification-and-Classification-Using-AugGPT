"""
=============================================================
SATD Replication  |  Script 1 of 3  —  Preprocessing
=============================================================
Paper : Deep Learning and Data Augmentation for Detecting
        Self-Admitted Technical Debt (arXiv:2410.15804)

Input : The 4 augmented CSVs from the replication package
        https://github.com/edisutoyo/satd-augmentation

        data-augmentation-code_comments.csv
        data-augmentation-issues.csv
        data-augmentation-pull-requests.csv
        data-augmentation-commit-messages.csv

        Each CSV has two columns:
            text   — raw text of the artifact
            class  — one of: C/D | DOC | TES | REQ | Not-SATD

Output: preprocessed_CC.csv  (and IS, PS, CM)
        Each output CSV has:
            text            — original text
            text_clean      — preprocessed text
            binary_label    — 0 = Not-SATD,  1 = SATD
            category_label  — 0=C/D 1=DOC 2=TES 3=REQ  (NaN for Not-SATD)

Usage:
    pip install pandas nltk scikit-learn
    python 01_preprocessing.py --data_dir ./data --out_dir ./processed

Section refs: III-D (Text Preprocessing), III-H (Dataset Split)
=============================================================
"""

import re
import string
import argparse
import os
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download required NLTK data once
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────

ARTIFACTS = {
    "CC": "data-augmentation-code_comments.csv",
    "IS": "data-augmentation-issues.csv",
    "PS": "data-augmentation-pull-requests.csv",
    "CM": "data-augmentation-commit-messages.csv",
}

SATD_TYPES = {"code_debt", "design_debt", "requirement_debt", "documentation_debt","test_debt"}

# Multi-class category labels (Section III-G)
CATEGORY_MAP = {"code_debt": 0, "documentation_debt": 1, "test_debt": 2, "requirement_debt": 3,"design_debt":4}

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()


# ── Text Preprocessing  (Section III-D) ──────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Applies all steps from Section III-D in order:
      1. Lowercase
      2. Remove URLs
      3. Remove non-ASCII characters
      4. Remove digits
      5. Remove punctuation
      6. Tokenise
      7. Remove stop words
      8. Remove short words (≤ 2 chars)
      9. Lemmatise
     10. Rejoin & strip extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                                          # 1
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # 2
    text = text.encode("ascii", errors="ignore").decode()        # 3
    text = re.sub(r"\d+", " ", text)                            # 4
    text = text.translate(                                       # 5
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    tokens = word_tokenize(text)                                 # 6
    tokens = [                                                   # 7-9
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)                                      # 10


# ── Label helpers ─────────────────────────────────────────────────────────────

def to_binary(cls: str) -> int:
    """SATD types → 1,  Not-SATD → 0."""
    return 1 if cls in SATD_TYPES else 0

def to_category(cls: str):
    """code_debt→0, documentation_debt→1, test_debt→2, requirement_debt→3, design_debt→4, Not-SATD→NaN."""
    return CATEGORY_MAP.get(cls, np.nan)


# ── Load one artifact CSV ─────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str, artifact: str) -> pd.DataFrame:
    """
    Load one augmented CSV, apply preprocessing, add label columns.
    """
    df = pd.read_csv(csv_path)
    assert {"text", "class"}.issubset(df.columns), \
        f"{csv_path} must contain 'text' and 'class' columns"

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["artifact"]       = artifact
    df["text_clean"]     = df["text"].apply(preprocess)
    df["binary_label"]   = df["class"].apply(to_binary)
    df["category_label"] = df["class"].apply(to_category)

    print(f"[{artifact}]  Total rows : {len(df)}")
    print(df["class"].value_counts().rename("count").to_string())
    print()
    return df


# ── Stratified 80 / 10 / 10 split  (Section III-H) ───────────────────────────

def split_80_10_10(df: pd.DataFrame, label_col: str, seed: int = 42):
    """
    Returns (train_df, val_df, test_df) with stratified sampling.
    Rows with NaN in label_col are dropped before splitting.
    """
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    train_val, test = train_test_split(
        df, test_size=0.10, stratify=df[label_col], random_state=seed
    )
    # 10 % of the full set ≈ 11.11 % of the remaining 90 %
    train, val = train_test_split(
        train_val, test_size=1/9, stratify=train_val[label_col], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for artifact, fname in ARTIFACTS.items():
        csv_path = os.path.join(data_dir, fname)
        if not os.path.exists(csv_path):
            print(f"⚠️  {csv_path} not found — skipping {artifact}\n")
            continue

        # 1. Load & preprocess
        df = load_and_preprocess(csv_path, artifact)

        # 2. Save full preprocessed file
        out_path = os.path.join(out_dir, f"preprocessed_{artifact}.csv")
        df.to_csv(out_path, index=False)

        # 3. Binary split (for BiLSTM identification)
        train, val, test = split_80_10_10(df, "binary_label")
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            split_df.to_csv(
                os.path.join(out_dir, f"{artifact}_binary_{split_name}.csv"),
                index=False,
            )
        print(f"  Binary split  → train:{len(train)}  val:{len(val)}  test:{len(test)}")

        # 4. Category split (for BERT — SATD rows only)
        satd_df = df[df["binary_label"] == 1].copy()
        train_c, val_c, test_c = split_80_10_10(satd_df, "category_label")
        for split_name, split_df in [("train", train_c), ("val", val_c), ("test", test_c)]:
            split_df.to_csv(
                os.path.join(out_dir, f"{artifact}_category_{split_name}.csv"),
                index=False,
            )
        print(f"  Category split → train:{len(train_c)}  val:{len(val_c)}  test:{len(test_c)}")
        print(f"  ✅  Saved to {out_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SATD Preprocessing — Script 1/3")
    parser.add_argument("--data_dir", default="./data",
                        help="Folder containing the 4 augmented CSVs")
    parser.add_argument("--out_dir",  default="./processed",
                        help="Folder to write preprocessed splits")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
