"""
pipeline.py – End-to-end runner for Subtask A.

Sequentially calls preprocess → train → evaluate.

Usage
-----
    python src/src_TaskA/pipeline.py [--skip_preprocess] [--skip_train]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent


def run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SRC_DIR / script)] + (extra_args or [])
    print(f"\n>>> Running: {' '.join(cmd)}\n{'─'*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Subtask A pipeline.")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if not args.skip_preprocess:
        run("preprocess.py")

    if not args.skip_train:
        run(
            "train.py",
            [
                "--model_name", args.model_name,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
            ],
        )

    run("evaluate.py")
    print("\nSubtask A pipeline complete.")


if __name__ == "__main__":
    main()
