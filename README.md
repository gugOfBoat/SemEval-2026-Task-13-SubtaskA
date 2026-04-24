# SemEval-2026 Task 13 — Subtask A: AI-Generated Code Detection

## Strategy

**Approach:** Gradient Boosting ensemble (LightGBM + XGBoost + CatBoost) on handcrafted language-agnostic features.

**Key insight:** Train set has 3 languages, test set has 8. Deep learning models overfit to language patterns. GBDT on structural/statistical features generalizes better.

---

## Pipeline

```
data/raw/Task_A/
  train.parquet + validation.parquet  →  merged training set
  test.parquet                        →  hidden test set
  test_sample.parquet                 →  1000 labeled samples for calibration

src/06_gbdt_ensemble.py
  [1] Load data
  [2] Extract 53 features (48 handcrafted + 5 compression)
  [3] TF-IDF (char 3-5 gram, 50k vocab) + SVD (200 dims)
  [4] Train LGB + XGB + CatBoost with StratifiedKFold-5
  [5] Grid-search ensemble weights + threshold on test_sample
  [6] Generate submission.csv
```

### Feature Groups (53 total)
| Group | Count | Examples |
|---|---|---|
| Line stats | 8 | avg/std/max line length, indentation stats |
| Whitespace | 2 | space ratio, tab ratio |
| Syntax | 9 | bracket counts, keyword ratio, comment ratio |
| Lexical | 10 | entropy, token diversity, identifier length |
| Style | 10 | snake_case, operator spacing, nesting depth |
| Patterns | 9 | duplicate lines, bigram repetition, import density |
| Compression | 5 | zlib ratio, gzip ratio, byte entropy |

---

## Kaggle Workflow

```python
# Cell 1 — Setup
!pip install lightgbm xgboost catboost scipy -q
%cd /kaggle/working
!rm -rf SemEval-2026-Task-13-SubtaskA
!git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

# Cell 2 — EDA (generates plots to /kaggle/working/eda_plots/)
!python SemEval-2026-Task-13-SubtaskA/src/01_eda.py

# Cell 3 — Run pipeline
!python SemEval-2026-Task-13-SubtaskA/src/06_gbdt_ensemble.py
```

Data path on Kaggle (read-only):
```
/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A/
  train.parquet
  validation.parquet
  test.parquet
  test_sample.parquet
  sample_submission.csv
```

---

## Data

```
data/
  raw/Task_A/       ← parquet files from Kaggle (not committed)
  download_data.py  ← Kaggle API download script
```

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` (see `.env.example`).

---

## Runtime

~60 minutes CPU-only on Kaggle (no GPU required).
Output: `submission.csv` in `/kaggle/working/` or `data/processed/v3/`.
