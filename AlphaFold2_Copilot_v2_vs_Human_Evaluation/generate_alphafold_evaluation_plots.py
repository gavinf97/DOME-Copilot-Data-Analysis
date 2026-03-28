#!/usr/bin/env python3
"""
Generate evaluation comparison plots for the AlphaFold2 single-paper Human vs Copilot run.

This script:
- Reads local evaluation results in this folder.
- Uses the prior 30-paper evaluation results as a canonical field reference.
- Plots outcomes across non-publication DOME fields.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

CURRENT_RESULTS = os.path.join(SCRIPT_DIR, "evaluation_results.tsv")
LEGACY_RESULTS = os.path.join(
    ROOT_DIR,
    "Human_30_Copilot_vs_Human_Evaluations_Interface",
    "evaluation_results.tsv",
)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Evaluation_Plots")

RANK_ORDER = ["A_Better", "Tie_High", "Tie_Low", "B_Better"]
RANK_COLORS = {
    "A_Better": "#1f77b4",
    "Tie_High": "#2ca02c",
    "Tie_Low": "#d62728",
    "B_Better": "#ff7f0e",
}
RANK_LABELS = {
    "A_Better": "Human better",
    "Tie_High": "Tie high quality",
    "Tie_Low": "Tie low quality",
    "B_Better": "Copilot v2 Better",
}
RANK_SCORE = {"A_Better": -1, "Tie_High": 0, "Tie_Low": 0, "B_Better": 1}


def load_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def canonical_non_publication_fields(legacy_df):
    legacy_valid = legacy_df.dropna(subset=["Field", "Rank"]).copy()
    legacy_valid = legacy_valid[~legacy_valid["Field"].astype(str).str.startswith("publication/")]
    legacy_valid = legacy_valid[legacy_valid["Rank"].isin(RANK_ORDER)]
    return sorted(legacy_valid["Field"].unique().tolist())


def prepare_current(current_df, allowed_fields):
    df = current_df.dropna(subset=["Field", "Rank"]).copy()
    df = df[df["Field"].isin(allowed_fields)]
    df = df[df["Rank"].isin(RANK_ORDER)]
    return df


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_field_outcomes(current_df):
    counts = (
        current_df.groupby(["Field", "Rank"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RANK_ORDER, fill_value=0)
    )

    if counts.empty:
        print("No rows available for field-outcome plot.")
        return

    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, max(7, 0.45 * len(counts))))
    left = None
    for rank in RANK_ORDER:
        vals = counts[rank].to_numpy()
        label = RANK_LABELS[rank]
        if left is None:
            ax.barh(counts.index, vals, color=RANK_COLORS[rank], label=label)
            left = vals
        else:
            ax.barh(counts.index, vals, left=left, color=RANK_COLORS[rank], label=label)
            left = left + vals

    ax.set_xlabel("Publication Count", fontsize=16, fontweight='bold')
    ax.set_ylabel("DOME Field", fontsize=16, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xlim(0, 1)
    ax.margins(y=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "01_Field_Outcome_Stacked.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_signed_field_comparison(current_df):
    df = current_df.copy()
    df["Score"] = df["Rank"].map(RANK_SCORE)

    summary = (
        df.groupby("Field", as_index=False)
        .agg(MeanScore=("Score", "mean"), N=("Score", "count"))
        .sort_values("MeanScore", ascending=True)
    )

    if summary.empty:
        print("No rows available for signed comparison plot.")
        return

    colors = ["#1f77b4" if x < 0 else "#ff7f0e" if x > 0 else "#7f7f7f" for x in summary["MeanScore"]]

    plt.figure(figsize=(12, max(7, 0.45 * len(summary))))
    bars = plt.barh(summary["Field"], summary["MeanScore"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlim(-1.05, 1.05)
    plt.xlabel("Comparison Score (-1 Human Better, 0 Tie, +1 Copilot Better)")
    plt.ylabel("Field")
    plt.title("AlphaFold2: Signed Field Comparison")

    for bar, n in zip(bars, summary["N"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        align = "left" if x >= 0 else "right"
        offset = 0.03 if x >= 0 else -0.03
        plt.text(x + offset, y, f"n={int(n)}", va="center", ha=align, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "02_Signed_Field_Comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


DISPLAY_ORDER = ["B_Better", "A_Better", "Tie_High", "Tie_Low"]


def plot_overall_distribution(current_df):
    counts = current_df["Rank"].value_counts().reindex(DISPLAY_ORDER, fill_value=0)
    if counts.sum() == 0:
        print("No rows available for overall distribution plot.")
        return

    display_labels = [RANK_LABELS[r] for r in counts.index]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(display_labels, counts.values, color=[RANK_COLORS[r] for r in counts.index])
    plt.xlabel("Manual Review Result", fontsize=16, fontweight='bold', labelpad=14)
    plt.ylabel("DOME Field Count", fontsize=16, fontweight='bold')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "03_Overall_Rank_Distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    sns.set_theme(style="whitegrid")
    ensure_plots_dir()

    legacy_df = load_tsv(LEGACY_RESULTS)
    current_df = load_tsv(CURRENT_RESULTS)

    allowed_fields = canonical_non_publication_fields(legacy_df)
    current_done = prepare_current(current_df, allowed_fields)

    if current_done.empty:
        raise RuntimeError(
            "No comparable non-publication field ratings found in current evaluation_results.tsv."
        )

    plot_field_outcomes(current_done)
    plot_signed_field_comparison(current_done)
    plot_overall_distribution(current_done)

    print("Done.")
    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Rows used: {len(current_done)}")
    print(f"Fields used: {current_done['Field'].nunique()}")


if __name__ == "__main__":
    main()
