"""
evaluate.py – Evaluate the fine-tuned Subtask A model on the test split.

Usage
-----
    python src/src_TaskA/evaluate.py [--model_dir <path>]

Outputs
-------
- Console: accuracy, macro-F1, full classification report
- File:    results/TaskA_results.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "TaskA" / "best"
LABEL_COL = "label"
TEXT_COL = "text"
BATCH_SIZE = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Subtask A model.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {args.model_dir}. "
            "Run src/src_TaskA/train.py first."
        )

    # Load test data
    test_path = DATA_DIR / "TaskA_test_processed.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test file not found: {test_path}. "
            "Run src/src_TaskA/preprocess.py first."
        )
    df = pd.read_parquet(test_path)

    # Run inference
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=128,
        batch_size=BATCH_SIZE,
    )

    texts = df[TEXT_COL].tolist()
    preds_raw = clf(texts)
    preds = [p["label"] for p in preds_raw]
    gold = df[LABEL_COL].tolist()

    # Metrics
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro")
    report = classification_report(gold, preds)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Save results
    results = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
    }
    out_path = RESULTS_DIR / "TaskA_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
