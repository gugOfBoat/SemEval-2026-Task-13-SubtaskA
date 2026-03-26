"""
info_dataset_subTaskA.py
Statistical and exploratory data analysis (EDA) for SemEval-2026 Task 13 – Subtask A.

Outputs
-------
- Console summary: shape, dtypes, missing values, class distribution
- Saved figures in:  img/img_TaskA/
    • class_distribution.png
    • text_length_distribution.png
    • missing_values_heatmap.png
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMG_DIR = REPO_ROOT / "img" / "img_TaskA"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Expected label column and text column names (adjust if dataset differs)
LABEL_COL = "label"
TEXT_COL = "text"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    """Load all parquet files from data/ that belong to Subtask A."""
    splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "dev", "test"):
        candidates = list(DATA_DIR.glob(f"*{split}*TaskA*.parquet")) + \
                     list(DATA_DIR.glob(f"*TaskA*{split}*.parquet")) + \
                     list(DATA_DIR.glob(f"*{split}*.parquet"))
        if candidates:
            path = candidates[0]
            splits[split] = pd.read_parquet(path)
            print(f"Loaded '{split}' split from {path.name}  ({len(splits[split])} rows)")
        else:
            print(f"[WARN] No parquet file found for split '{split}' – skipping.")
    return splits


def print_summary(df: pd.DataFrame, split_name: str) -> None:
    print(f"\n{'='*60}")
    print(f" Split: {split_name}  |  Shape: {df.shape}")
    print(f"{'='*60}")
    print(df.dtypes.to_string())
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
    if LABEL_COL in df.columns:
        print(f"\nClass distribution ({LABEL_COL}):\n{df[LABEL_COL].value_counts().to_string()}")


def plot_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    if LABEL_COL not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df[LABEL_COL].value_counts().index
    sns.countplot(data=df, x=LABEL_COL, order=order, ax=ax, palette="viridis")
    ax.set_title(f"Class Distribution – {split_name}")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = IMG_DIR / f"class_distribution_{split_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_text_length_distribution(df: pd.DataFrame, split_name: str) -> None:
    if TEXT_COL not in df.columns:
        return
    lengths = df[TEXT_COL].dropna().str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="white")
    ax.set_title(f"Text Length Distribution (words) – {split_name}")
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    out = IMG_DIR / f"text_length_distribution_{split_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_missing_values(df: pd.DataFrame, split_name: str) -> None:
    missing = df.isnull()
    if not missing.values.any():
        print(f"[{split_name}] No missing values – skipping heatmap.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(missing, cbar=False, yticklabels=False, ax=ax, cmap="viridis")
    ax.set_title(f"Missing Values Heatmap – {split_name}")
    plt.tight_layout()
    out = IMG_DIR / f"missing_values_heatmap_{split_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    splits = load_data()

    if not splits:
        print(
            "\nNo data files found in 'data/'. "
            "Run 'python data.py' first to download the dataset."
        )
        return

    for split_name, df in splits.items():
        print_summary(df, split_name)
        plot_class_distribution(df, split_name)
        plot_text_length_distribution(df, split_name)
        plot_missing_values(df, split_name)

    print(f"\nAll plots saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
