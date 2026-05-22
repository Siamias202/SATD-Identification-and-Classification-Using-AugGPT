"""
=============================================================
General SATD Preprocessing (Single Merged Dataset)
=============================================================

Input CSV columns:
    text
    class

Classes:
    non_debt
    code_debt
    design_debt
    documentation_debt
    test_debt
    requirement_debt

Outputs:
    processed/
        satd_processed.csv

        binary_train.csv
        binary_val.csv
        binary_test.csv

        category_train.csv
        category_val.csv
        category_test.csv
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

# ============================================================
# NLTK resources
# ============================================================

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ============================================================
# Labels
# ============================================================

SATD_TYPES = {
    "code_debt",
    "design_debt",
    "documentation_debt",
    "test_debt",
    "requirement_debt",
}

CATEGORY_MAP = {
    "non_debt": 0,
    "code_debt": 1,
    "documentation_debt": 2,
    "test_debt": 3,
    "requirement_debt": 4,
    "design_debt": 5,
}

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ============================================================
# Preprocessing
# ============================================================

def preprocess_glove(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # remove non-ascii
    text = text.encode("ascii", errors="ignore").decode()

    # remove digits
    text = re.sub(r"\d+", " ", text)

    # remove punctuation
    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords + short words + lemmatize
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return " ".join(tokens)

def preprocess_llm_embeddings(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove only URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # keep punctuation (IMPORTANT for code/LLMs)
    # remove only excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================================================
# Labels
# ============================================================

def to_binary(label: str) -> int:
    return 1 if label in SATD_TYPES else 0

def to_category(label: str):
    return CATEGORY_MAP.get(label, np.nan)

# ============================================================
# Split
# ============================================================

def split_80_10_10(df, label_col, seed=42):

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    train_val, test = train_test_split(
        df,
        test_size=0.10,
        stratify=df[label_col],
        random_state=seed
    )

    train, val = train_test_split(
        train_val,
        test_size=1/9,
        stratify=train_val[label_col],
        random_state=seed
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True)
    )

# ============================================================
# Main
# ============================================================

def main(input_csv, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    # ========================================================
    # Load dataset
    # ========================================================

    df = pd.read_csv(input_csv)

    required_cols = {"text", "class"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: text, class"
        )

    # remove missing rows
    df = df.dropna(subset=["text"])

    # remove duplicate text rows
    df = df.drop_duplicates(subset=["text"])

    df = df.reset_index(drop=True)

    print("\nDataset loaded")
    print(f"Rows: {len(df)}")

    print("\nClass distribution:")
    print(df["class"].value_counts())

    # ========================================================
    # Preprocess text
    # ========================================================

    print("\nPreprocessing text...")

    df["text_clean"] = df["text"].apply(preprocess_glove)

    # ========================================================
    # Labels
    # ========================================================

    df["binary_label"] = df["class"].apply(to_binary)

    df["category_label"] = df["class"].apply(to_category)

    # ========================================================
    # Save full processed dataset
    # ========================================================

    full_output = os.path.join(
        out_dir,
        "satd_processed.csv"
    )

    df.to_csv(full_output, index=False)

    print(f"\nSaved full dataset:")
    print(full_output)

    # ========================================================
    # Binary split
    # ========================================================

    print("\nCreating binary splits...")

    train_b, val_b, test_b = split_80_10_10(
        df,
        "binary_label"
    )

    train_b.to_csv(
        os.path.join(out_dir, "binary_train.csv"),
        index=False
    )

    val_b.to_csv(
        os.path.join(out_dir, "binary_val.csv"),
        index=False
    )

    test_b.to_csv(
        os.path.join(out_dir, "binary_test.csv"),
        index=False
    )

    # ========================================================
    # Multiclass split
    # ========================================================

    print("Creating multiclass splits...")

    satd_df = df[df["binary_label"] == 1].copy()

    train_c, val_c, test_c = split_80_10_10(
        satd_df,
        "category_label"
    )

    train_c.to_csv(
        os.path.join(out_dir, "category_train.csv"),
        index=False
    )

    val_c.to_csv(
        os.path.join(out_dir, "category_val.csv"),
        index=False
    )

    test_c.to_csv(
        os.path.join(out_dir, "category_test.csv"),
        index=False
    )

    print("\nDONE")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        required=True,
        help="Merged SATD CSV file"
    )

    parser.add_argument(
        "--out_dir",
        default="./processed",
        help="Output directory"
    )

    args = parser.parse_args()

    main(args.input_csv, args.out_dir)