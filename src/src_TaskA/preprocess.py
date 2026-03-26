"""
preprocess.py – Data preprocessing pipeline for Subtask A.

Reads raw parquet files from data/, cleans and tokenizes the text,
and writes processed outputs back to data/ with a '_processed' suffix.
"""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
LABEL_COL = "label"
TEXT_COL = "text"


def clean_text(text: str) -> str:
    """Basic text cleaning: strip whitespace and normalise."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def preprocess_split(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    """Clean text and add token-count column for diagnostic purposes."""
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].map(clean_text)
    df["token_count"] = df[TEXT_COL].apply(
        lambda t: len(tokenizer.tokenize(t, truncation=True, max_length=MAX_LENGTH))
    )
    return df


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split in ("train", "dev", "test"):
        candidates = list(DATA_DIR.glob(f"*{split}*TaskA*.parquet")) + \
                     list(DATA_DIR.glob(f"*TaskA*{split}*.parquet")) + \
                     list(DATA_DIR.glob(f"*{split}*.parquet"))
        if not candidates:
            print(f"[WARN] No file found for split '{split}' – skipping.")
            continue

        path = candidates[0]
        df = pd.read_parquet(path)
        df_processed = preprocess_split(df, tokenizer)

        out_path = DATA_DIR / f"TaskA_{split}_processed.parquet"
        df_processed.to_parquet(out_path, index=False)
        print(f"Saved processed '{split}' split → {out_path.name}")


if __name__ == "__main__":
    main()
