# CAMSP v10 — Compression-Aware Meta-Stacking Pipeline

> **SemEval 2026 Task 13 Subtask A**: AI-Generated Code Detection  
> *Detecting machine-generated code across 8+ programming languages with Out-of-Distribution resilience*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Academic-green)

---

## Executive Summary

**CAMSP** treats AI code detection as a **Code Forensics** problem.  Instead of relying on vocabulary-based features that break on unseen programming languages, CAMSP exploits the fundamental insight that **AI-generated code is structurally over-regular**: it compresses better, has more uniform indentation, and produces lower perplexity under language models.

### Out-of-Distribution (OOD) Resilience

The test set (500k samples) contains **unseen languages** not present in training. CAMSP's defense strategy:

| Signal | Why It Works on OOD |
|--------|---------------------|
| Compression ratios (zlib, bz2) | Language-agnostic: measures byte-level regularity |
| Shannon byte entropy | Captures information density regardless of syntax |
| Indent delta entropy | AI models produce mechanically consistent spacing |
| LLM perplexity (Qwen-0.5B) | Neural fingerprint of "how expected" the code is |
| Adaptive ratio shrinkage | Prevents collapse on low-confidence OOD subsets |

---

## Methodology — Four Pillars

### 1. Stacking Ensemble (4 Base Estimators)

```
char_full    ─┐
char_family  ─┤── 5-Fold OOF ──► HGB Meta-Learner ──► Final Score
word_hash    ─┤
style_hgb    ─┘
```

- **char_full**: Char-level (3,6)-gram TF-IDF + SGD logistic regression
- **char_family**: Same architecture, but trained with inverse-sqrt family weights to balance generator diversity
- **word_hash**: Word (1,3)-gram hashing vectorizer (2^20 features) + SGD
- **style_hgb**: 15+ compression/entropy features fed into HistGradientBoosting

### 2. LLM Perplexity Engine (Test-First Strategy)

Uses **Qwen2.5-Coder-0.5B** quantized to **NF4 4-bit** (BitsAndBytes) to compute token-level negative log-likelihood.

**Key innovation**: Budget allocation prioritizes the test set (55% of time → ~25-30% test coverage) before consuming remaining budget on sample and train subsets.

### 3. Adaptive Constraint Engine (OODRatioTuner)

Prevents the "ratio collapse" failure mode where the model labels everything as human on OOD data:

- **Global ratio floor**: Clamped to `[0.10, 0.40]` (never below 10%)
- **Per-language tuning**: Independent ratios per language on `test_sample.parquet`
- **Shrinkage interpolation**: `ratio = (1-s) * global + s * per_lang` with `s ∈ {0, 0.25, 0.5, 0.75, 1.0}`

### 4. Extended Compression Features

Beyond standard zlib ratio, CAMSP adds:
- `bz2_ratio` — Burrows-Wheeler block-sorting compression
- `byte_entropy` — Shannon entropy over raw byte distribution
- `indent_delta_entropy` — Entropy of indentation changes between lines
- `line_len_cv` — Coefficient of variation of line lengths
- `trigram_rep_ratio` — Character trigram repetition rate

---

## Repository Structure

```
SemEval-2026-Task-13-SubtaskA/
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # PipelineConfig dataclass (all hyperparams)
│   ├── data_utils.py         # DataIngestion, ArtifactDetector, GeneratorFamilyEncoder
│   ├── features.py           # CodeStyleExtractor, LLMPerplexityEngine
│   ├── tuning.py             # OODRatioTuner (adaptive shrinkage)
│   └── orchestrator.py       # CAMSPipeline (end-to-end runner)
├── scripts/
│   └── run_inference.py      # Kaggle entrypoint
└── README.md
```

---

## Kaggle Setup Guide

### Prerequisites
| Setting | Value |
|---------|-------|
| **GPU** | T4 x2 (recommended) or P100 |
| **Internet** | ON (to clone repo) |
| **Persistence** | Files only |

### Input Data (Already on Kaggle)
1. **Competition data**: `semeval-2026-task13-subtask-a` → auto-discovered via recursive walk of `/kaggle/input/`
2. **Model** (optional): Add `Qwen/Qwen2.5-Coder` → version `0.5b-instruct` from Kaggle Models tab. If not added, the script downloads it automatically via HuggingFace.

### Run (Single Cell)

```python
!pip install bitsandbytes -q

%cd /kaggle/working
!rm -rf SemEval-2026-Task-13-SubtaskA
!git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

%cd SemEval-2026-Task-13-SubtaskA
!python scripts/run_inference.py
```

### Time Budget (~4 hours total, fits 12h limit)

| Phase | Duration | Description |
|-------|----------|-------------|
| LLM Perplexity | ~70 min | Qwen 0.5B NF4 on test+sample+train |
| Style Features | ~18 min | Compression/entropy extraction |
| 5-Fold Stacking | ~150 min | 4 base models × 5 folds |
| Meta + Tuning | ~2 min | HGB stacking + ratio search |

### VRAM & Speed Notes
- Qwen-0.5B at NF4 4-bit uses **~500MB VRAM** — stable on T4 (16GB)
- Batch size 64 with 64-token sequences → **~60 samples/sec** throughput
- All sparse matrices (TF-IDF) use CSR format — peak RAM ~8GB

---

## Results

| Version | Sample F1 | Leaderboard | Key Change |
|---------|-----------|-------------|------------|
| v6 (baseline) | 0.679 | ~0.65 | Weighted blend |
| v9 (stacking) | 0.719 | ~0.68 | K-fold stacking + LLM perplexity |
| **v10 (CAMSP)** | **TBD** | **Target: 0.70** | Test-first LLM ordering, +5 features, ratio floor |

---

## License

Academic use only — SemEval 2026 competition submission.
