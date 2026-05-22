# SATD Replication — 3-Script Pipeline

Replication of **"Deep Learning and Data Augmentation for Detecting
Self-Admitted Technical Debt"** (Sutoyo, Avgeriou, Capiluppi — arXiv:2410.15804)

---

## What each script does

| Script | Task | Model | Input | Output |
|--------|------|-------|-------|--------|
| `01_preprocessing.py` | Clean text, add labels, split 80/10/10 | — | 4 augmented CSVs | `processed/` folder |
| `02_bilstm_identification.py` | Binary: SATD vs Not-SATD | BiLSTM + GloVe | `*_binary_*.csv` | `bilstm_{ART}_best.pt` |
| `03_bert_categorization.py` | Multi-class: C/D / DOC / TES / REQ | BERT-base | `*_category_*.csv` | `bert_{ART}_best.pt` |

---

## Setup

```bash
pip install torch transformers pandas numpy scikit-learn nltk
```

Optional (for best BiLSTM performance):
```bash
# Download GloVe 100-d vectors (~820 MB unzipped)
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Keep only: glove.6B.100d.txt
```

---

## Step 0 — Get the data

Download the 4 augmented CSVs from the authors' replication package:
```
https://github.com/edisutoyo/satd-augmentation
```

Put them in a `data/` folder:
```
data/
  data-augmentation-code_comments.csv
  data-augmentation-issues.csv
  data-augmentation-pull-requests.csv
  data-augmentation-commit-messages.csv
```

---

## Step 1 — Preprocessing

```bash
python 01_preprocessing.py --data_dir ./data --out_dir ./processed
```

Produces for each artifact (CC, IS, PS, CM):
```
processed/
  preprocessed_CC.csv          ← full cleaned dataset
  CC_binary_train/val/test.csv ← for BiLSTM  (all rows)
  CC_category_train/val/test.csv ← for BERT  (SATD rows only)
```

---

## Step 2 — BiLSTM Identification

Run for each artifact separately:

```bash
# With GloVe (recommended)
python 02_bilstm_identification.py \
    --processed_dir ./processed \
    --artifact CC \
    --glove_path ./glove.6B.100d.txt

# Without GloVe (random embeddings, lower performance)
python 02_bilstm_identification.py \
    --processed_dir ./processed \
    --artifact CC
```

Expected results (Table V of the paper):

| Artifact | Not-SATD F1 | SATD F1 | Macro-avg F1 |
|----------|-------------|---------|--------------|
| CC | 0.952 | 0.927 | **0.939** |
| IS | 0.937 | 0.820 | **0.878** |
| PS | 0.917 | 0.806 | **0.862** |
| CM | 0.940 | 0.821 | **0.880** |

---

## Step 3 — BERT Categorization

```bash
python 03_bert_categorization.py \
    --processed_dir ./processed \
    --artifact CC
```

Expected results (Table VI of the paper):

| Artifact | C/D F1 | DOC F1 | TES F1 | REQ F1 | Macro-avg F1 |
|----------|--------|--------|--------|--------|--------------|
| CC | 0.885 | 0.925 | 0.925 | 0.796 | **0.882** |
| IS | 0.902 | 0.922 | 0.922 | 0.851 | **0.899** |
| PS | 0.842 | 0.895 | 0.851 | 0.842 | **0.876** |
| CM | 0.882 | 0.826 | 0.841 | 0.840 | **0.847** |

### Predict a single text after training:

```bash
python 03_bert_categorization.py \
    --artifact CC \
    --predict "TODO: this is a temporary hack, need to fix later"
# Output: Predicted SATD type: C/D
```

---

## Run all 4 artifacts in one go

```bash
for ART in CC IS PS CM; do
  echo "=== $ART ==="
  python 02_bilstm_identification.py --processed_dir ./processed \
      --artifact $ART --glove_path ./glove.6B.100d.txt
  python 03_bert_categorization.py --processed_dir ./processed \
      --artifact $ART
done
```

---

## Key hyperparameters

| Parameter | BiLSTM | BERT |
|-----------|--------|------|
| Optimiser | Adam | AdamW |
| Learning rate | 1e-3 | **5e-5** |
| Batch size | 32 | **32** |
| Max sequence length | 128 | 128 |
| Dropout | 0.3 | 0.1 |
| Early stopping (patience) | 5 epochs | 3 epochs |
| Stop criterion | val loss | val loss |
| AdamW epsilon | — | **1e-8** |

---

## GPU note

Both scripts auto-detect a CUDA GPU. On CPU, BERT training will be slow
(~1–2 hrs per artifact). On a single V100 (as used in the paper) expect
~5–10 min per artifact per script.
