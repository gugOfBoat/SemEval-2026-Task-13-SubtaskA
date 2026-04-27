"""
CAMSP v10 — Feature Engineering Engines.

Two complementary feature extraction systems:
1. CodeStyleExtractor: Language-agnostic compression and structural metrics.
2. LLMPerplexityEngine: Test-first neural perplexity with strict time budgets.
"""

import bz2
import gc
import logging
import os
import subprocess
import sys
import time
import zlib
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class CodeStyleExtractor:
    """Extracts compression-aware stylometric features from source code.

    Produces language-agnostic signals that capture the structural
    regularity of AI-generated code vs human-written code:
    - Compression ratios (zlib, bz2)
    - Byte-level Shannon entropy
    - Indentation delta entropy
    - Line statistics

    Args:
        config: Pipeline configuration with ``max_chars`` attribute.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def _extract_single(self, code: str) -> dict:
        """Computes style features for a single code sample.

        Args:
            code: Raw source code string.

        Returns:
            Dictionary of feature name -> float value.
        """
        if not isinstance(code, str) or len(code) == 0:
            return {}

        lines = code.split("\n")
        non_empty = [line for line in lines if line.strip()]
        f = {
            "line_count": max(len(lines), 1),
            "char_count": max(len(code), 1),
        }
        f["empty_line_ratio"] = 1.0 - (len(non_empty) / f["line_count"])

        # --- Line length statistics ---
        ll = np.array([len(line) for line in lines], dtype=np.float32)
        f["avg_line_length"] = float(ll.mean())
        f["std_line_length"] = float(ll.std())
        f["max_line_length"] = float(ll.max())
        f["line_len_cv"] = float(ll.std()) / max(float(ll.mean()), 1e-6)

        # --- Compression ratios ---
        tb = code[: self.cfg.max_chars].encode("utf-8", errors="replace")
        blen = max(len(tb), 1)
        f["zlib_ratio"] = len(zlib.compress(tb, level=1)) / blen if tb else 0.0

        if tb:
            f["bz2_ratio"] = len(bz2.compress(tb, compresslevel=9)) / blen
            byte_arr = np.frombuffer(tb, dtype=np.uint8)
            cnts = np.bincount(byte_arr, minlength=256)
            probs = cnts[cnts > 0] / byte_arr.size
            f["byte_entropy"] = float(-(probs * np.log2(probs)).sum())
        else:
            f["bz2_ratio"] = 0.0
            f["byte_entropy"] = 0.0

        # --- Indentation dynamics ---
        indents = [len(line) - len(line.lstrip()) for line in lines]
        if non_empty:
            ne_ind = np.array(
                [len(line) - len(line.lstrip()) for line in non_empty],
                dtype=np.float32,
            )
            f["indent_std"] = float(ne_ind.std())
            f["indent_unique"] = float(len(set(ne_ind.tolist())))
        else:
            f["indent_std"] = 0.0
            f["indent_unique"] = 0.0

        deltas = [abs(indents[i + 1] - indents[i]) for i in range(len(indents) - 1)]
        if deltas:
            dc = Counter(deltas)
            dt = sum(dc.values())
            pd_ = np.array(list(dc.values()), dtype=np.float64) / dt
            f["indent_delta_entropy"] = float(-(pd_ * np.log2(pd_ + 1e-12)).sum())
        else:
            f["indent_delta_entropy"] = 0.0

        # --- Character distribution ---
        cc = max(len(code), 1)
        char_counter = Counter(code)
        cp = np.array(list(char_counter.values()), dtype=np.float64) / cc
        f["char_entropy"] = float(-np.sum(cp * np.log2(cp + 1e-12)))
        f["unique_char_ratio"] = len(char_counter) / cc
        f["space_ratio"] = code.count(" ") / cc
        f["newline_ratio"] = code.count("\n") / cc

        # --- Code smell indicators ---
        f["has_markdown_fence"] = int("```" in code)
        f["has_special_token"] = int("\x3c|" in code)

        # --- Trigram repetition ---
        if len(code) >= 3:
            trigrams = [code[i : i + 3] for i in range(len(code) - 2)]
            tc = Counter(trigrams)
            f["trigram_rep_ratio"] = sum(1 for c in tc.values() if c > 1) / max(
                len(tc), 1
            )
        else:
            f["trigram_rep_ratio"] = 0.0

        return f

    def extract_batch(self, codes: np.ndarray, desc: str) -> pd.DataFrame:
        """Extracts style features for an entire array of code samples.

        Args:
            codes: NumPy array of raw code strings.
            desc: Human-readable label for progress logging.

        Returns:
            DataFrame with one row per sample, columns = feature names.
        """
        logger.info("Extracting CAMSP style features for %s (%d samples)", desc, len(codes))
        t0 = time.time()
        rows = []
        for i, code in enumerate(codes, 1):
            try:
                rows.append(self._extract_single(code))
            except Exception:
                rows.append({})
            if i % 100_000 == 0:
                logger.info("  %d / %d | %.0f it/s", i, len(codes), i / max(time.time() - t0, 1))
        df = pd.DataFrame(rows).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        logger.info("%s: done in %.1fs | shape=%s", desc, time.time() - t0, df.shape)
        return df


class LLMPerplexityEngine:
    """Computes token-level NLL features using a quantized causal LM.

    Implements **Test-First Strategy**: allocates 55% of the time budget
    to the test set before processing sample and train subsets. This
    maximizes perplexity coverage on the evaluation target.

    Falls back to zero features if no GPU or transformers is available.

    Args:
        config: Pipeline configuration with LLM parameters.
    """

    FEATURE_NAMES = ["nll_mean", "nll_std", "nll_q25", "nll_q75", "token_count"]

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def execute(
        self,
        train_codes: np.ndarray,
        test_codes: np.ndarray,
        sample_codes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Runs the full perplexity pipeline with test-first budget allocation.

        Args:
            train_codes: Training code array.
            test_codes: Test code array (500k samples).
            sample_codes: Optional test_sample code array.

        Returns:
            Tuple of (ppl_train, ppl_test, ppl_sample) feature matrices.
        """
        logger.info("Initializing LLM Perplexity Engine (Test-First Strategy)")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            logger.warning("transformers unavailable — returning zero features")
            return (
                self._zeros(len(train_codes)),
                self._zeros(len(test_codes)),
                self._zeros(len(sample_codes)) if sample_codes is not None else None,
            )

        model, tokenizer = self._load_model()
        if model is None:
            return (
                self._zeros(len(train_codes)),
                self._zeros(len(test_codes)),
                self._zeros(len(sample_codes)) if sample_codes is not None else None,
            )

        t0 = time.time()
        budget = self.cfg.ppl_time_budget_sec

        # Priority 1: TEST (55% of budget)
        logger.info("Phase 1/3: Test set LLM fingerprinting")
        ppl_test, n_test = self._infer(test_codes, model, tokenizer, budget * 0.55)
        logger.info("Test LLM coverage: %d / %d (%.1f%%)", n_test, len(test_codes), n_test / len(test_codes) * 100)

        # Priority 2: SAMPLE (small, fast)
        ppl_sample = None
        if sample_codes is not None:
            remaining = budget - (time.time() - t0)
            logger.info("Phase 2/3: Sample set LLM fingerprinting")
            ppl_sample, _ = self._infer(sample_codes, model, tokenizer, min(remaining, 120))

        # Priority 3: TRAIN subsample (whatever time is left)
        remaining = budget - (time.time() - t0)
        ppl_train = self._zeros(len(train_codes))
        if remaining > 120:
            logger.info("Phase 3/3: Train subsample LLM fingerprinting")
            n_sub = min(self.cfg.ppl_train_subsample, len(train_codes))
            sub_idx = np.sort(np.random.choice(len(train_codes), n_sub, replace=False))
            ppl_sub, n_done = self._infer(train_codes[sub_idx], model, tokenizer, remaining)
            ppl_train[sub_idx[:n_done]] = ppl_sub[:n_done]

        del model, tokenizer
        import torch; torch.cuda.empty_cache()
        gc.collect()
        logger.info("LLM Perplexity done in %.1f min", (time.time() - t0) / 60)
        return ppl_train, ppl_test, ppl_sample

    def _load_model(self):
        """Attempts to load a quantized causal LM from Kaggle model inputs."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("No CUDA available")
                return None, None
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            return None, None

        for path in self.cfg.ppl_candidates:
            if path.startswith("/") and not os.path.isdir(path):
                continue
            try:
                logger.info("Trying LLM: %s", path)
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="right")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.eval()
                logger.info("Loaded %s (BnB NF4 4-bit)", path)
                return model, tokenizer
            except Exception as exc:
                logger.warning("Failed %s: %s", path, exc)
        return None, None

    def _infer(self, codes, model, tokenizer, time_budget: float):
        """Batch LLM inference with graceful time-budget abort."""
        import torch

        n = len(codes)
        features = self._zeros(n)
        bs = self.cfg.ppl_batch_size
        t0 = time.time()
        last_end = 0

        for start in range(0, n, bs):
            end = min(start + bs, n)
            last_end = end
            batch = [c[: self.cfg.max_chars] if isinstance(c, str) else "" for c in codes[start:end]]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=self.cfg.ppl_max_tokens, padding=True)
            ids = enc.input_ids.to(model.device)
            mask = enc.attention_mask.to(model.device)

            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask).logits

            sl = logits[:, :-1, :].contiguous()
            st = ids[:, 1:].contiguous()
            sm = mask[:, 1:].contiguous().float()
            nll = (
                torch.nn.CrossEntropyLoss(reduction="none")(
                    sl.view(-1, sl.size(-1)), st.view(-1)
                ).view(st.size())
                * sm
            )

            for j in range(end - start):
                vals = nll[j][sm[j].bool()].float().cpu().numpy()
                if len(vals) == 0:
                    continue
                q25, q75 = np.percentile(vals, [25, 75])
                features[start + j] = [
                    np.mean(vals), np.std(vals), q25, q75, float(len(vals)),
                ]

            del ids, mask, logits, sl, st, sm, nll
            torch.cuda.empty_cache()

            if time.time() - t0 > time_budget:
                logger.info("Time budget reached at %d / %d", end, n)
                break

        return features, last_end

    def _zeros(self, n: int) -> np.ndarray:
        return np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float32)
