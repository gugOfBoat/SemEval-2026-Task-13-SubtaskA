"""
CAMSP v10 — Pipeline Orchestrator.

Wires together data ingestion, feature extraction, stacking,
meta-learning, and adaptive ratio tuning into a single callable pipeline.
"""

import gc
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import PipelineConfig
from .data_utils import ArtifactDetector, DataIngestion, GeneratorFamilyEncoder, set_seed
from .features import CodeStyleExtractor, LLMPerplexityEngine
from .tuning import OODRatioTuner

logger = logging.getLogger(__name__)


class CAMSPipeline:
    """Compression-Aware Meta-Stacking Pipeline orchestrator.

    Execution flow:
        1. Auto-discover and load parquet splits.
        2. Extract LLM perplexity features (test-first budget).
        3. Extract compression-aware style features.
        4. K-fold stacking with 4 base estimators.
        5. HGB meta-learner on OOF predictions + perplexity.
        6. Adaptive ratio tuning on test_sample.
        7. Generate submission.csv.

    Args:
        config: Pipeline configuration. Uses defaults if None.
    """

    def __init__(self, config: PipelineConfig = None) -> None:
        self.cfg = config or PipelineConfig()
        self.data_mgr = DataIngestion(self.cfg)
        self.style_eng = CodeStyleExtractor(self.cfg)
        self.ppl_eng = LLMPerplexityEngine(self.cfg)
        self.tuner = OODRatioTuner(self.cfg)

    def _truncate(self, codes) -> list:
        """Truncates code samples to max_chars for vectorization."""
        return [str(x)[: self.cfg.max_chars] if x else "" for x in codes]

    def run(self) -> pd.DataFrame:
        """Executes the full CAMSP pipeline end-to-end.

        Returns:
            Submission DataFrame with columns [ID, label].
        """
        set_seed(self.cfg.seed)
        t_start = time.time()

        # ── 1. Data Loading ──
        tr_df, va_df, te_df, sa_df = self.data_mgr.load_splits()
        tr_full = pd.concat([tr_df, va_df], ignore_index=True)
        del tr_df, va_df
        gc.collect()
        logger.info("Train+Val combined: %d samples", len(tr_full))

        y_train = tr_full["label"].astype(int).values
        fw_train = GeneratorFamilyEncoder.build_weights(tr_full["generator"])

        te_artifacts = ArtifactDetector.detect(te_df["code"].values, self.cfg.special_tokens)
        sa_artifacts = (
            ArtifactDetector.detect(sa_df["code"].values, self.cfg.special_tokens)
            if sa_df is not None else None
        )

        # ── 2. LLM Perplexity (Test-First) ──
        ppl_tr, ppl_te, ppl_sa = self.ppl_eng.execute(
            tr_full["code"].values,
            te_df["code"].values,
            sa_df["code"].values if sa_df is not None else None,
        )

        # ── 3. Style Features ──
        sty_tr = self.style_eng.extract_batch(tr_full["code"].values, "Train")
        sty_te = self.style_eng.extract_batch(te_df["code"].values, "Test")
        sty_sa = (
            self.style_eng.extract_batch(sa_df["code"].values, "Sample")
            if sa_df is not None else None
        )

        # Merge perplexity into style
        for k, col in enumerate(LLMPerplexityEngine.FEATURE_NAMES):
            sty_tr[f"ppl_{col}"] = ppl_tr[:, k]
            sty_te[f"ppl_{col}"] = ppl_te[:, k]
            if sty_sa is not None and ppl_sa is not None:
                sty_sa[f"ppl_{col}"] = ppl_sa[:, k]

        X_sty_all = sty_tr.astype(np.float32).values
        X_sty_te = sty_te.astype(np.float32).values
        X_sty_sa = sty_sa.astype(np.float32).values if sty_sa is not None else None
        del sty_tr, sty_te, sty_sa
        gc.collect()

        # ── 4. K-Fold Stacking ──
        n_train, n_test = len(tr_full), len(te_df)
        n_sample = len(sa_df) if sa_df is not None else 0

        oof = np.zeros((n_train, 4), dtype=np.float32)
        te_sum = np.zeros((n_test, 4), dtype=np.float32)
        sa_sum = np.zeros((n_sample, 4), dtype=np.float32) if n_sample > 0 else None

        # Pre-fit char vocabulary
        logger.info("Pre-computing char vocabulary")
        cv_master = TfidfVectorizer(
            analyzer="char", ngram_range=self.cfg.char_ngram_range,
            max_features=self.cfg.char_max_features, min_df=3,
            sublinear_tf=True, lowercase=False, dtype=np.float32,
        )
        cv_master.fit(self._truncate(tr_full["code"].values))
        char_vocab = cv_master.vocabulary_
        del cv_master
        gc.collect()

        wv = HashingVectorizer(
            analyzer="word", token_pattern=r"\b\w+\b", ngram_range=(1, 3),
            n_features=self.cfg.word_hash_features, alternate_sign=False,
            lowercase=False, norm="l2", dtype=np.float32,
        )

        skf = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
            t_fold = time.time()
            logger.info("Fold %d/%d (train=%d, val=%d)", fi + 1, self.cfg.n_folds, len(tr_idx), len(va_idx))

            y_tr = y_train[tr_idx]
            fw_tr = fw_train[tr_idx]
            fold_codes = tr_full.iloc[tr_idx]["code"].values

            # Base 1+2: Char TF-IDF
            cv = TfidfVectorizer(
                analyzer="char", ngram_range=self.cfg.char_ngram_range,
                vocabulary=char_vocab, sublinear_tf=True, lowercase=False, dtype=np.float32,
            )
            Xct = cv.fit_transform(self._truncate(fold_codes))
            Xcv = cv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xce = cv.transform(self._truncate(te_df["code"].values))
            Xcs = cv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c1 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c1.fit(Xct, y_tr)
            oof[va_idx, 0] = c1.decision_function(Xcv).astype(np.float32)
            te_sum[:, 0] += c1.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 0] += c1.decision_function(Xcs).astype(np.float32)

            c2 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha * 1.5, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c2.fit(Xct, y_tr, sample_weight=fw_tr)
            oof[va_idx, 1] = c2.decision_function(Xcv).astype(np.float32)
            te_sum[:, 1] += c2.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 1] += c2.decision_function(Xcs).astype(np.float32)
            del Xct, Xcv, Xce, Xcs, c1, c2, cv
            gc.collect()

            # Base 3: Word hash
            Xwt = wv.transform(self._truncate(fold_codes))
            Xwv = wv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xwe = wv.transform(self._truncate(te_df["code"].values))
            Xws = wv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c3 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c3.fit(Xwt, y_tr)
            oof[va_idx, 2] = c3.decision_function(Xwv).astype(np.float32)
            te_sum[:, 2] += c3.decision_function(Xwe).astype(np.float32)
            if Xws is not None:
                sa_sum[:, 2] += c3.decision_function(Xws).astype(np.float32)
            del Xwt, Xwv, Xwe, Xws, c3
            gc.collect()

            # Base 4: Style HGB
            Xs_tr, ys_tr = X_sty_all[tr_idx], y_tr
            if len(Xs_tr) > self.cfg.style_subsample:
                si = np.random.choice(len(Xs_tr), self.cfg.style_subsample, replace=False)
                Xs_tr, ys_tr = Xs_tr[si], y_tr[si]

            c4 = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=63,
                min_samples_leaf=40, l2_regularization=0.1, random_state=self.cfg.seed,
            )
            c4.fit(Xs_tr, ys_tr)
            oof[va_idx, 3] = c4.predict_proba(X_sty_all[va_idx])[:, 1].astype(np.float32)
            te_sum[:, 3] += c4.predict_proba(X_sty_te)[:, 1].astype(np.float32)
            if X_sty_sa is not None:
                sa_sum[:, 3] += c4.predict_proba(X_sty_sa)[:, 1].astype(np.float32)
            del c4
            gc.collect()
            logger.info("Fold %d done in %.1fs", fi + 1, time.time() - t_fold)

        te_avg = te_sum / self.cfg.n_folds
        sa_avg = sa_sum / self.cfg.n_folds if sa_sum is not None else None
        del X_sty_all, X_sty_te, X_sty_sa
        gc.collect()

        # ── 5. Meta-Learner ──
        logger.info("Training HGB meta-learner")
        Xm_tr = np.column_stack([oof, ppl_tr])
        Xm_te = np.column_stack([te_avg, ppl_te])
        Xm_sa = np.column_stack([sa_avg, ppl_sa]) if sa_avg is not None and ppl_sa is not None else None

        meta = HistGradientBoostingClassifier(
            learning_rate=self.cfg.meta_lr, max_iter=self.cfg.meta_max_iter,
            max_leaf_nodes=self.cfg.meta_max_leaf_nodes, min_samples_leaf=50,
            l2_regularization=1.0, random_state=self.cfg.seed,
        )
        meta.fit(Xm_tr, y_train)
        meta_te = meta.predict_proba(Xm_te)[:, 1].astype(np.float32)
        meta_sa = meta.predict_proba(Xm_sa)[:, 1].astype(np.float32) if Xm_sa is not None else None
        del meta, Xm_tr, Xm_te, Xm_sa, oof
        gc.collect()

        # ── 6. Adaptive Ratio Tuning ──
        te_langs = te_df["language"].fillna("Unknown").astype(str).values

        if sa_df is not None and meta_sa is not None:
            sa_langs = sa_df["language"].fillna("Unknown").astype(str)
            tune_cfg = self.tuner.tune(sa_df["label"].values, meta_sa, sa_langs, sa_artifacts)
        else:
            tune_cfg = {"global": self.cfg.fallback_global_ratio, "l_map": {}, "shrink": 0.5}

        norm_scores = self.tuner.rank_normalize(meta_te)
        preds = self.tuner.language_aware_predict(
            norm_scores, te_langs, tune_cfg["global"], tune_cfg["l_map"], tune_cfg["shrink"],
        )
        preds[te_artifacts] = 1

        # ── 7. Save Submission ──
        id_col = "ID" if "ID" in te_df.columns else "id"
        sub = pd.DataFrame({"ID": te_df[id_col].values, "label": preds.astype(int)})

        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        out_path = os.path.join(out_dir, "submission.csv")
        sub.to_csv(out_path, index=False)

        logger.info("Submission saved: %s", out_path)
        logger.info("Machine ratio: %.2f%% (%d / %d)", sub["label"].mean() * 100, sub["label"].sum(), len(sub))

        # Evaluate on sample if available
        if sa_df is not None and meta_sa is not None:
            sa_norm = self.tuner.rank_normalize(meta_sa)
            sa_preds = self.tuner.language_aware_predict(
                sa_norm, sa_langs.values, tune_cfg["global"], tune_cfg["l_map"], tune_cfg["shrink"],
            )
            sa_preds[sa_artifacts] = 1
            logger.info("Sample F1: %.4f", f1_score(sa_df["label"].values, sa_preds, average="macro"))

        logger.info("Total elapsed: %.1f minutes", (time.time() - t_start) / 60)
        return sub
