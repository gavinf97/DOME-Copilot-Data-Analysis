#!/usr/bin/env python3
"""
generate_average_field_length_panels.py

Compare the average response length per DOME field between the Copilot v0 and
v2 positive 1012 datasets, excluding publication metadata fields.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_V0 = os.path.join(
    SCRIPT_DIR,
    "../Copilot_Processed_Datasets_JSON/Copilot_1012_v0_Processed_2026-01-15_Updated_Metadata",
)
DATASET_V2 = os.path.join(
    SCRIPT_DIR,
    "../Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Pos_Processed_2026-03-02_Updated_Metadata",
)
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Graph_Panel_Field_Lengths")

CATEGORY_MAP = {
    "dataset": "Data",
    "optimization": "Optimisation",
    "model": "Model",
    "evaluation": "Evaluation",
}

VERSION_STYLES = {
    "v0": {"label": "Copilot v0", "color": "#7EB1DD"},
    "v2": {"label": "Copilot v2", "color": "#27AE60"},
}


def response_length(value):
    if value is None:
        return 0

    if isinstance(value, str):
        return len(value.strip())

    return len(str(value).strip())


def load_average_lengths(dataset_dir, version_name):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Data folder not found: {dataset_dir}")

    json_files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_dir}")

    records = []

    print(f"Processing {len(json_files)} files from {dataset_dir}")
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for field_name, value in data.items():
            if "/" not in field_name:
                continue

            prefix, subfield = field_name.split("/", 1)
            if prefix == "publication" or prefix not in CATEGORY_MAP:
                continue

            records.append(
                {
                    "Version": version_name,
                    "Category": CATEGORY_MAP[prefix],
                    "Subfield": subfield,
                    "Length": response_length(value),
                }
            )

    frame = pd.DataFrame(records)
    grouped = (
        frame.groupby(["Version", "Category", "Subfield"], as_index=False)["Length"]
        .mean()
        .rename(columns={"Length": "AverageLength"})
    )
    return grouped


def build_comparison_frame():
    v0_frame = load_average_lengths(DATASET_V0, "v0")
    v2_frame = load_average_lengths(DATASET_V2, "v2")

    combined = pd.concat([v0_frame, v2_frame], ignore_index=True)
    return combined


def format_subfield_label(subfield_name):
    return subfield_name.replace("_", " ").capitalize()


def create_comparison_plot(comparison_df):
    import matplotlib.patches as mpatches
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Overall averages per version for legend annotation
    v0_global_avg = comparison_df[comparison_df["Version"] == "v0"]["AverageLength"].mean()
    v2_global_avg = comparison_df[comparison_df["Version"] == "v2"]["AverageLength"].mean()

    categories = ["Data", "Optimisation", "Model", "Evaluation"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(
        "Average Response Length per Field: Copilot v0 vs v2",
        fontsize=28,
        fontweight="bold",
        y=1.02,
    )
    axes = axes.flatten()

    for index, category in enumerate(categories):
        ax = axes[index]
        category_df = comparison_df[comparison_df["Category"] == category].copy()

        pivot_df = (
            category_df.pivot(index="Subfield", columns="Version", values="AverageLength")
            .fillna(0)
            .reset_index()
        )

        if "v0" not in pivot_df.columns:
            pivot_df["v0"] = 0
        if "v2" not in pivot_df.columns:
            pivot_df["v2"] = 0

        pivot_df["Label"] = pivot_df["Subfield"].apply(format_subfield_label)
        pivot_df["SortLength"] = pivot_df[["v0", "v2"]].max(axis=1)
        pivot_df = pivot_df.sort_values("SortLength", ascending=True)

        y_positions = np.arange(len(pivot_df))
        bar_height = 0.36

        # v0 on top, v2 below
        v0_bars = ax.barh(
            y_positions + bar_height / 2,
            pivot_df["v0"],
            bar_height,
            color=VERSION_STYLES["v0"]["color"],
            label=VERSION_STYLES["v0"]["label"],
        )
        v2_bars = ax.barh(
            y_positions - bar_height / 2,
            pivot_df["v2"],
            bar_height,
            color=VERSION_STYLES["v2"]["color"],
            label=VERSION_STYLES["v2"]["label"],
        )

        max_value = max(pivot_df[["v0", "v2"]].max().max(), 1)
        ax.set_xlim(0, max_value * 1.30)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(pivot_df["Label"])
        ax.set_title(category, fontweight="bold", fontsize=24)
        ax.set_xlabel("Average response length (characters)", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)

        for bar_version, bars in [("v0", v0_bars), ("v2", v2_bars)]:
            for bar in bars:
                width = bar.get_width()
                if width <= 0:
                    continue
                # Nudge v2 labels down slightly in the Optimisation panel to
                # clear the v0 bar above them
                y_offset = -bar_height * 0.35 if (category == "Optimisation" and bar_version == "v2") else 0
                ax.text(
                    width + (max_value * 0.025),
                    bar.get_y() + (bar.get_height() / 2) + y_offset,
                    f"{width:.0f}",
                    ha="left",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    color="black",
                )

    v0_patch = mpatches.Patch(
        color=VERSION_STYLES["v0"]["color"],
        label=f"Copilot v0  (overall avg: {v0_global_avg:.0f} chars)",
    )
    v2_patch = mpatches.Patch(
        color=VERSION_STYLES["v2"]["color"],
        label=f"Copilot v2  (overall avg: {v2_global_avg:.0f} chars)",
    )
    total_patch = mpatches.Patch(color="black", label="Avg. length shown at bar end")

    fig.legend(
        handles=[v0_patch, v2_patch, total_patch],
        loc="upper right",
        fontsize=16,
        bbox_to_anchor=(0.985, 0.985),
        framealpha=0.95,
        edgecolor="black",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.5)

    output_png = os.path.join(OUTPUT_FOLDER, "average_field_length_v0_vs_v2.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_png}")


def main():
    comparison_df = build_comparison_frame()
    create_comparison_plot(comparison_df)


if __name__ == "__main__":
    main()