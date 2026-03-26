"""
train.py – Fine-tune a transformer model for Subtask A (text classification).

Usage
-----
    python src/src_TaskA/train.py [options]

Key options
-----------
--model_name   Hugging Face model identifier   (default: bert-base-uncased)
--epochs       Number of training epochs        (default: 3)
--batch_size   Per-device training batch size   (default: 16)
--lr           Learning rate                    (default: 2e-5)
--output_dir   Where to save the fine-tuned model
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "TaskA"

LABEL_COL = "label"
TEXT_COL = "text"
MAX_LENGTH = 128


def load_split(split: str) -> pd.DataFrame:
    """Load the processed parquet for a given split."""
    path = DATA_DIR / f"TaskA_{split}_processed.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run src/src_TaskA/preprocess.py first."
        )
    return pd.read_parquet(path)


def build_dataset(tokenizer: AutoTokenizer, label2id: dict) -> DatasetDict:
    """Build a HuggingFace DatasetDict from parquet files."""
    splits = {}
    for split in ("train", "dev"):
        df = load_split(split)
        df["labels"] = df[LABEL_COL].map(label2id)
        hf_ds = Dataset.from_pandas(df[[TEXT_COL, "labels"]], preserve_index=False)
        hf_ds = hf_ds.map(
            lambda batch: tokenizer(
                batch[TEXT_COL],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )
        hf_ds = hf_ds.remove_columns([TEXT_COL])
        splits[split] = hf_ds
    return DatasetDict(splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Subtask A classifier.")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Build label mapping from training data
    train_df = load_split("train")
    labels = sorted(train_df[LABEL_COL].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = build_dataset(tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(args.output_dir / "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    tokenizer.save_pretrained(str(args.output_dir / "best"))
    print(f"Model saved to {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
