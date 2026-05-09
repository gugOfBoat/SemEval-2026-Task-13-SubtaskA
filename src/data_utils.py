"""
CAMSP v10 — Data Ingestion & Encoding Utilities.

Handles Kaggle dataset auto-discovery, parquet loading, artifact
detection, and generator family weighting for balanced training.
"""

import logging
import math
import os
import random
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets global random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class DataIngestion:
    """Discovers and loads SemEval Task 13A parquet datasets.

    Probes multiple Kaggle input directory layouts to locate the
    train, validation, test and test_sample splits automatically.

    Args:
        config: Pipeline configuration instance.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def discover_data_directory(self) -> str:
        """Recursively searches for the dataset directory.

        Returns:
            Absolute path to the directory containing parquet files.

        Raises:
            FileNotFoundError: If no valid dataset directory is found.
        """
        candidates = [
            "/kaggle/input/semeval-2026-task13-subtask-a/Task_A",
            "/kaggle/input/SemEval-2026-Task13-Subtask-A/Task_A",
            "/kaggle/input/semeval-2026-task13-subtask-a/task_A",
            "/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A",
            "/kaggle/input/semeval-2026-task13/task_A",
            "/kaggle/input/semeval-2026-task13/Task_A",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        if os.path.exists("/kaggle/input"):
            for dirpath, _, filenames in os.walk("/kaggle/input"):
                if "train.parquet" in filenames:
                    return dirpath

        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        if os.path.exists(os.path.join(local_path, "train.parquet")):
            return local_path

        raise FileNotFoundError(
            "Could not auto-discover dataset. "
            "Add the competition data as a Kaggle input source or place it in the ../data/ directory."
        )

    def load_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Loads all dataset splits from discovered directory.

        Returns:
            Tuple of (train_df, val_df, test_df, sample_df).
            sample_df is None if test_sample.parquet does not exist.
        """
        data_dir = self.discover_data_directory()
        self.config.data_dir = data_dir
        logger.info("Dataset directory: %s", data_dir)

        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "validation.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        sample_path = os.path.join(data_dir, "test_sample.parquet")
        sample_df = (
            pd.read_parquet(sample_path) if os.path.exists(sample_path) else None
        )

        logger.info(
            "Loaded -> Train: %s | Val: %s | Test: %s | Sample: %s",
            f"{len(train_df):,}",
            f"{len(val_df):,}",
            f"{len(test_df):,}",
            f"{len(sample_df):,}" if sample_df is not None else "N/A",
        )
        return train_df, val_df, test_df, sample_df


class ArtifactDetector:
    """Identifies code samples containing LLM generation artifacts.

    Detects leaked special tokens, markdown fences, and conversational
    preambles that indicate the code was copy-pasted from a chatbot.
    """

    @staticmethod
    def detect(codes: np.ndarray, special_tokens: List[str]) -> np.ndarray:
        """Scans code array for hard artifact indicators.

        Args:
            codes: Array of raw code strings.
            special_tokens: LLM control tokens to search for.

        Returns:
            Boolean mask where True = detected artifact.
        """
        results = np.zeros(len(codes), dtype=bool)
        for i, code in enumerate(codes):
            if not isinstance(code, str):
                continue
            if any(tok in code for tok in special_tokens):
                results[i] = True
            elif "```" in code:
                results[i] = True
            elif re.match(
                r"^(Here is|Here's|Sure,|Certainly|Below is|The following)", code
            ):
                results[i] = True
        logger.info(
            "Artifacts: %d / %d (%.2f%%)",
            results.sum(),
            len(codes),
            results.mean() * 100,
        )
        return results


class GeneratorFamilyEncoder:
    """Maps generator model names to families and builds training weights.

    AI models from the same family produce similar code patterns.
    Inverse-sqrt weighting prevents dominant families from overwhelming
    the classifier.
    """

    _FAMILY_MAP = [
        ("phi", "phi"),
        ("qwen", "qwen"),
        ("llama", "llama"),
        ("gemma", "gemma"),
        ("gpt", "gpt"),
        ("deepseek", "deepseek"),
        ("yi-coder", "yi"),
        ("starcoder", "starcoder"),
        ("codegemma", "gemma"),
        ("codellama", "llama"),
        ("mistral", "mistral"),
        ("claude", "claude"),
        ("command-r", "command-r"),
        ("mixtral", "mistral"),
    ]

    @classmethod
    def normalize(cls, name: str) -> str:
        """Reduces a model identifier to its canonical family name."""
        if not isinstance(name, str):
            return "unknown"
        lowered = name.lower()
        if lowered == "human":
            return "human"
        for needle, family in cls._FAMILY_MAP:
            if needle in lowered:
                return family
        return lowered.split("/", 1)[0] if "/" in lowered else lowered.split("-", 1)[0]

    @classmethod
    def build_weights(cls, generator_col: pd.Series) -> np.ndarray:
        """Computes inverse-sqrt frequency weights per generator family.

        Args:
            generator_col: Series of generator identifiers.

        Returns:
            Float32 weight array normalized to mean=1.0.
        """
        families = generator_col.map(cls.normalize)
        counts = families.value_counts().to_dict()
        weights = (
            families.map(lambda x: 1.0 / math.sqrt(counts[x]))
            .astype(np.float32)
            .values
        )
        weights *= len(weights) / weights.sum()
        return weights


class FamilyInferencer:
    """Infers programming language family from code syntax patterns.

    Groups code into broad families (python, c_style, scripting, unknown)
    to enable family-aware ratio tuning when the 'language' column is absent.
    """

    _RULES = [
        ('python', [
            (re.compile(r'^\s*def\s+\w+\s*\(', re.M), 3),
            (re.compile(r'^\s*(import|from)\s+\w+', re.M), 2),
            (re.compile(r'^\s*elif\s+', re.M), 3),
            (re.compile(r'^\s*class\s+\w+.*:', re.M), 2),
            (re.compile(r':\s*$', re.M), 1),
        ]),
        ('c_style', [
            (re.compile(r'#include\s*[<"]', re.M), 4),
            (re.compile(r'\b(int|void|char|double|float)\s+\w+\s*[\(;]'), 3),
            (re.compile(r'\bstruct\s+\w+\s*\{'), 3),
            (re.compile(r'^\s*\}', re.M), 1),
            (re.compile(r'\bpackage\s+\w+'), 2),
            (re.compile(r'\bpublic\s+(static\s+)?class\b'), 3),
        ]),
        ('scripting', [
            (re.compile(r'\bfunction\s+\w+\s*\('), 3),
            (re.compile(r'\bconsole\.(log|error|warn)\b'), 3),
            (re.compile(r'^#!/', re.M), 4),
            (re.compile(r'\bconst\s+\w+\s*=\s*\(.*\)\s*=>'), 3),
            (re.compile(r'<\?php'), 4),
            (re.compile(r'\$\w+\s*='), 2),
        ]),
    ]

    @classmethod
    def infer(cls, code: str) -> str:
        """Infers language family for a single code sample."""
        if not isinstance(code, str) or len(code) < 10:
            return 'unknown'
        head = '\n'.join(code.split('\n')[:50])
        scores = {}
        for family, patterns in cls._RULES:
            scores[family] = sum(w for rx, w in patterns if rx.search(head))
        best_fam = max(scores, key=scores.get)
        return best_fam if scores[best_fam] >= 3 else 'unknown'

    @classmethod
    def infer_batch(cls, codes: np.ndarray) -> np.ndarray:
        """Infers language families for an array of code samples."""
        logger.info("Inferring language families for %d samples", len(codes))
        t0 = time.time()
        result = np.array([cls.infer(c) for c in codes], dtype=object)
        vals, counts = np.unique(result, return_counts=True)
        dist = dict(zip(vals, counts))
        logger.info("Family distribution: %s (%.1fs)", dist, time.time() - t0)
        return result
