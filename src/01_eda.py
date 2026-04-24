#!/usr/bin/env python3
"""
SemEval-2026 Task 13A — EDA
============================
Chạy trên Kaggle notebook (dùng %run để plots hiện inline):

    Cell 1:
        !pip install scipy -q
        %cd /kaggle/working
        !git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

    Cell 2:
        %matplotlib inline
        %run SemEval-2026-Task-13-SubtaskA/src/01_eda.py
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A")
OUT_DIR  = Path("/kaggle/working/eda_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

C_BLUE   = "#4C9BE8"
C_RED    = "#E8634C"
C_GREEN  = "#56B17B"
C_YELLOW = "#F0C040"
LABEL_PALETTE = {0: C_BLUE, 1: C_RED}
LANG_PALETTE  = [C_BLUE, C_RED, C_GREEN, C_YELLOW,
                 "#9B5DE5", "#F15BB5", "#00BBF9", "#FF9F1C"]


def show(title: str = ""):
    """Tight layout + plt.show() — displays inline when run with %run."""
    plt.tight_layout()
    plt.show()


def divider(title: str):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
divider("1 · Load data")

train_df  = pd.read_parquet(DATA_DIR / "train.parquet")
val_df    = pd.read_parquet(DATA_DIR / "validation.parquet")
ts_df     = pd.read_parquet(DATA_DIR / "test_sample.parquet")
test_df   = pd.read_parquet(DATA_DIR / "test.parquet")

# Merge train + val → full training set
tv_df = pd.concat([train_df, val_df], ignore_index=True)

# Normalise label: some splits use 'Human'/'AI' strings
def _norm_label(df):
    df = df.copy()
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"human": 0, "ai": 1})
    return df

tv_df  = _norm_label(tv_df)
ts_df  = _norm_label(ts_df)

for name, df in [("train (raw)",  train_df),
                  ("val   (raw)",  val_df),
                  ("train+val",    tv_df),
                  ("test_sample",  ts_df),
                  ("test",         test_df)]:
    ncol = len(df.columns)
    lang = sorted(df["language"].unique()) if "language" in df.columns else "—"
    print(f"  {name:14s}: {len(df):>8,} rows | {ncol} cols | langs={lang}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAMPLE ROWS — mặt mũi dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
divider("2 · Sample rows")

# Show 3 rows per split, truncate code to 120 chars
def show_sample(df, name, n=3):
    print(f"\n  ── {name} (first {n} rows) ──")
    sample = df.head(n).copy()
    if "code" in sample.columns:
        sample["code_preview"] = sample["code"].str[:120].str.replace("\n", "↵")
    cols_to_show = [c for c in ["ID","language","label","generator","code_preview"]
                    if c in sample.columns]
    print(sample[cols_to_show].to_string(index=False))

show_sample(tv_df,   "train+val")
show_sample(ts_df,   "test_sample")
show_sample(test_df, "test (no label)")

# Code length stats
for name, df in [("train+val", tv_df), ("test_sample", ts_df)]:
    lens = df["code"].str.len()
    print(f"\n  code length [{name}]: "
          f"min={lens.min()} | median={int(lens.median())} | "
          f"mean={int(lens.mean())} | max={lens.max():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL DISTRIBUTION (pie + bar)
# ─────────────────────────────────────────────────────────────────────────────
divider("3 · Label distribution per split")

label_map = {0: "Human", 1: "AI"}

splits_labeled = {
    "train (raw)":  _norm_label(train_df),
    "val   (raw)":  _norm_label(val_df),
    "train+val":    tv_df,
    "test_sample":  ts_df,
}

print("\n  Label counts & AI rate:")
for name, df in splits_labeled.items():
    vc = df["label"].value_counts().sort_index()
    ai_r = df["label"].mean()
    print(f"  {name:14s}: Human={vc.get(0,0):>8,}  AI={vc.get(1,0):>8,}  "
          f"AI_rate={ai_r:.2%}")

# Figure: 2×2 pie charts
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Label Distribution per Split", fontsize=14, fontweight="bold", y=1.02)

for ax, (name, df) in zip(axes, splits_labeled.items()):
    vc = df["label"].value_counts().sort_index()
    labels = [label_map[i] for i in vc.index]
    colors = [LABEL_PALETTE[i] for i in vc.index]
    wedges, texts, auts = ax.pie(
        vc.values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.8,
        wedgeprops=dict(width=0.55, edgecolor="white"),
    )
    for a in auts:
        a.set_fontsize(10)
    total = vc.sum()
    ax.set_title(f"{name}\n(n={total:,})", fontsize=11, fontweight="bold")

show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. LANGUAGE DISTRIBUTION — THE OOD PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
divider("4 · Language distribution — OOD shift")

train_langs = set(tv_df["language"].unique())
test_langs  = set(ts_df["language"].unique())
unseen      = test_langs - train_langs

print(f"\n  Train+Val languages ({len(train_langs)}): {sorted(train_langs)}")
print(f"  Test Sample languages ({len(test_langs)}): {sorted(test_langs)}")
print(f"  ⚠  Unseen in training ({len(unseen)}): {sorted(unseen)}")
print(f"\n  Train+Val: {len(tv_df):,} rows — "
      f"{tv_df['language'].value_counts(normalize=True).mul(100).round(1).to_dict()}")
print(f"  Test Sample: {len(ts_df)} rows — "
      f"{ts_df['language'].value_counts().to_dict()}")

# Figure: side-by-side grouped bars (language proportion)
all_langs  = sorted(test_langs | train_langs)
tv_pct = tv_df["language"].value_counts(normalize=True).reindex(all_langs, fill_value=0) * 100
ts_pct = ts_df["language"].value_counts(normalize=True).reindex(all_langs, fill_value=0) * 100

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(all_langs))
w = 0.38
b1 = ax.bar(x - w/2, tv_pct.values, w, label="Train+Val", color=C_BLUE,  edgecolor="white")
b2 = ax.bar(x + w/2, ts_pct.values, w, label="Test Sample (OOD proxy)",
            color=C_RED, edgecolor="white", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(all_langs, rotation=25, ha="right")
ax.set_ylabel("Proportion (%)")
ax.set_title("Language Distribution: Train+Val vs Test Sample\n"
             "⚠  Red = languages UNSEEN in training (OOD shift)",
             fontweight="bold")
ax.legend()

# Mark unseen languages
for i, lang in enumerate(all_langs):
    if lang in unseen:
        ax.annotate("UNSEEN", xy=(x[i] + w/2, ts_pct[lang] + 0.3),
                    ha="center", va="bottom", fontsize=7.5,
                    color=C_RED, fontweight="bold")

show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. AI RATE PER LANGUAGE × SPLIT
# ─────────────────────────────────────────────────────────────────────────────
divider("5 · AI rate per language")

combined = pd.concat([
    tv_df.assign(split="Train+Val"),
    ts_df.assign(split="Test Sample"),
])

ai_rate = (combined.groupby(["split", "language"])["label"]
           .agg(ai_rate="mean", n="count")
           .reset_index())

print("\n  AI rate per language × split:")
pivot = ai_rate.pivot(index="language", columns="split", values="ai_rate").fillna(0)
for lang in pivot.index:
    row = pivot.loc[lang]
    tag = " ← UNSEEN" if lang in unseen else ""
    print(f"  {lang:12s}: Train+Val={row.get('Train+Val',0):.2%}  "
          f"Test Sample={row.get('Test Sample',0):.2%}{tag}")

# Figure: grouped bar plot
fig, ax = plt.subplots(figsize=(11, 4))

tv_rate = ai_rate[ai_rate["split"] == "Train+Val"].set_index("language")["ai_rate"]
ts_rate = ai_rate[ai_rate["split"] == "Test Sample"].set_index("language")["ai_rate"]
ts_n    = ai_rate[ai_rate["split"] == "Test Sample"].set_index("language")["n"]

x = np.arange(len(all_langs))
ax.bar(x - w/2,
       [tv_rate.get(l, 0) for l in all_langs], w,
       label="Train+Val", color=C_BLUE, edgecolor="white")
ax.bar(x + w/2,
       [ts_rate.get(l, 0) for l in all_langs], w,
       label="Test Sample", color=C_RED, edgecolor="white", alpha=0.9)

# Annotate n on test sample bars
for i, lang in enumerate(all_langs):
    n = ts_n.get(lang, 0)
    if n > 0:
        ax.text(x[i] + w/2, ts_rate.get(lang, 0) + 0.01, f"n={n}",
                ha="center", fontsize=7.5, color="gray")

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% line")
ax.set_xticks(x)
ax.set_xticklabels(all_langs, rotation=25, ha="right")
ax.set_ylabel("AI-generated rate")
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("AI Rate per Language × Split\n"
             "Test sample: only 22% AI — very different from training distribution",
             fontweight="bold")
ax.legend()
show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. CODE LENGTH DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
divider("6 · Code length distribution")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Code Length Distribution (chars) by Label", fontweight="bold")

caps = {0: "Human", 1: "AI"}
for ax, (name, df) in zip(axes, [("Train+Val", tv_df), ("Test Sample", ts_df)]):
    for label, color in [(0, C_BLUE), (1, C_RED)]:
        vals = df[df["label"] == label]["code"].str.len().clip(upper=5000)
        vals.plot.hist(ax=ax, bins=60, alpha=0.55, color=color,
                       label=caps[label], edgecolor="none")
    ax.set_xlabel("Code length (chars, clipped at 5000)")
    ax.set_ylabel("Count")
    ax.set_title(name)
    ax.legend()

show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. GENERATOR BREAKDOWN (train+val only)
# ─────────────────────────────────────────────────────────────────────────────
divider("7 · Generator breakdown (train+val)")

gen_vc = tv_df["generator"].value_counts()
print(f"\n  Top 15 generators:\n{gen_vc.head(15).to_string()}")
print(f"\n  Total unique generators: {gen_vc.shape[0]}")

top_gen = gen_vc.head(12).reset_index()
top_gen.columns = ["generator", "count"]

fig, ax = plt.subplots(figsize=(10, 4))
colors = [C_BLUE if g == "human" else C_RED for g in top_gen["generator"]]
ax.barh(top_gen["generator"].str.replace("human", "Human (label=0)"),
        top_gen["count"], color=colors, edgecolor="white")
ax.set_xlabel("Count")
ax.set_title("Top 12 Generators in Train+Val\n"
             "Blue = human, Red = AI models", fontweight="bold")
ax.invert_yaxis()

ax.legend(handles=[mpatches.Patch(color=C_BLUE, label="Human"),
                   mpatches.Patch(color=C_RED,  label="AI model")])
show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY DASHBOARD — OOD AT A GLANCE
# ─────────────────────────────────────────────────────────────────────────────
divider("8 · OOD Summary Dashboard")

fig = plt.figure(figsize=(14, 7))
fig.suptitle("SemEval-2026 Task 13A — Dataset OOD Summary",
             fontsize=14, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

# ── Panel A: Language counts (stacked bar) ───────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
ai_counts  = combined[combined["label"] == 1].groupby(["split", "language"])["label"].count().unstack(fill_value=0)
hum_counts = combined[combined["label"] == 0].groupby(["split", "language"])["label"].count().unstack(fill_value=0)

bar_langs = sorted(combined["language"].unique())
for split_name, color, hatch in [("Train+Val", C_BLUE, ""), ("Test Sample", C_RED, "//")]:
    sub_ai  = ai_counts.loc[split_name]  if split_name in ai_counts.index  else pd.Series(dtype=float)
    sub_hum = hum_counts.loc[split_name] if split_name in hum_counts.index else pd.Series(dtype=float)
    bot = 0
    for lang in bar_langs:
        h_val = sub_hum.get(lang, 0)
        a_val = sub_ai.get(lang, 0)
        if h_val + a_val == 0:
            continue
        ax_a.barh(split_name, h_val, left=bot, color=color, alpha=0.6, label="_" if lang != bar_langs[0] else split_name)
        ax_a.barh(split_name, a_val, left=bot + h_val, color=color, alpha=0.9, hatch=hatch)
        bot += h_val + a_val

ax_a.set_xlabel("Number of samples")
ax_a.set_title("Sample counts (solid=Human, hatched=AI)")
ax_a.legend(handles=[mpatches.Patch(color=C_BLUE, label="Train+Val"),
                      mpatches.Patch(color=C_RED, label="Test Sample")],
            loc="lower right")

# ── Panel B: Language proportion donut ───────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
ts_lvc = ts_df["language"].value_counts()
lang_colors_b = LANG_PALETTE[:len(ts_lvc)]
wedges, texts, auts = ax_b.pie(
    ts_lvc.values, labels=ts_lvc.index, autopct="%1.0f%%",
    colors=lang_colors_b, startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor="white"),
)
for t in auts: t.set_fontsize(7.5)
for t in texts: t.set_fontsize(8.5)
ax_b.set_title("Test Sample\nlanguage mix", fontweight="bold")

# ── Panel C: AI rate bars ────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, :])
ts_ai_r = ts_df.groupby("language")["label"].mean().sort_values(ascending=False)
tv_ai_r = tv_df.groupby("language")["label"].mean()
all_lang_c = ts_ai_r.index.tolist()
x_c = np.arange(len(all_lang_c))
w_c = 0.35

ax_c.bar(x_c - w_c/2, [tv_ai_r.get(l, 0) for l in all_lang_c], w_c,
         label="Train+Val", color=C_BLUE, edgecolor="white")
ax_c.bar(x_c + w_c/2, ts_ai_r.values, w_c,
         label="Test Sample", color=C_RED, edgecolor="white", alpha=0.9)
ax_c.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels(all_lang_c, rotation=20, ha="right")
ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax_c.set_ylabel("AI rate")
ax_c.set_title("AI Rate per Language (Train+Val vs Test Sample)\n"
               "Bars without blue = UNSEEN language in training ⚠")
ax_c.legend()

show()


# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
divider("EDA Complete — Key Takeaways")

print(f"""
  DATA OVERVIEW
  ─────────────
  Train+Val : {len(tv_df):>8,} rows | langs: {sorted(train_langs)} | AI rate: {tv_df['label'].mean():.2%}
  Test Sample:  1,000 rows | langs: {sorted(test_langs)} | AI rate: {ts_df['label'].mean():.2%}
  Test      : {len(test_df):>8,} rows | no labels

  OOD SHIFT
  ─────────
  Training is {tv_df['language'].value_counts(normalize=True)['Python']*100:.1f}% Python.
  Test sample has {len(test_langs)} languages — {len(unseen)} completely UNSEEN: {sorted(unseen)}

  CLASS SHIFT
  ───────────
  Train AI rate: {tv_df['label'].mean():.2%}  →  Test sample AI rate: {ts_df['label'].mean():.2%}
  Model must NOT rely on class prior — explicit threshold calibration needed.

  IMPLICATION
  ───────────
  Any feature encoding language-specific syntax (TF-IDF char n-grams, 
  semicolons, tab vs space) will fail on unseen languages.
  Need: features that are STABLE across all 8 languages.

""")
