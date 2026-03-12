#!/usr/bin/env python3
"""
generate_graph_panels_negative.py

Generate negative-dataset coverage/yield panels for the v2 negative 1012 set.
For negative papers, a field is considered successful if it either:
1. Returns "Not a relevant AI or machine learning publication"
2. Returns only "Not enough information" placeholders and no other content

Any additional generated content is treated as failure.
"""

import glob
import json
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(
    SCRIPT_DIR,
    "../Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Neg_Processed_2026-03-02_Updated_Metadata",
)
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Graph_Panel_V2_Neg")

CATEGORIES_MAP = {
    "dataset": "Data",
    "optimization": "Optimisation",
    "model": "Model",
    "evaluation": "Evaluation",
}

IRRELEVANT_PHRASE = "not a relevant ai or machine learning publication"
MISSING_PATTERN = re.compile(r"not enough information(?: is)?(?: available)?")


def strip_to_alnum(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_response_text(value_str):
    text = value_str.lower().replace("\r", "\n")
    text = text.replace("**", "")

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[\*\-\u2022]+\s*", "", line)
        line = re.sub(r"\s+", " ", line)

        prev_line = None
        while prev_line != line:
            prev_line = line
            line = re.sub(r"^[^:\n]{1,80}:\s*", "", line).strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def classify_negative_response(value_str):
    if not isinstance(value_str, str) or not value_str.strip():
        return "Empty"

    normalized = normalize_response_text(value_str)
    if not normalized:
        return "Empty"

    irrelevant_removed = normalized.replace(IRRELEVANT_PHRASE, " ")
    if IRRELEVANT_PHRASE in normalized and not strip_to_alnum(irrelevant_removed):
        return "IrrelevantSuccess"

    allowed_removed = normalized.replace(IRRELEVANT_PHRASE, " ")
    allowed_removed = MISSING_PATTERN.sub(" ", allowed_removed)

    if not strip_to_alnum(allowed_removed):
        if MISSING_PATTERN.search(normalized):
            return "NotEnoughOnly"
        return "Empty"

    if MISSING_PATTERN.search(normalized):
        return "MixedFailure"

    return "Failure"


def load_and_process_data():
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder not found at {DATA_FOLDER}")
        return None

    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    if not json_files:
        print("No files found. Exiting.")
        return None

    results = []
    print(f"Found {len(json_files)} JSON files in {DATA_FOLDER}")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            for key, value in data.items():
                if "/" not in key:
                    continue

                prefix, subfield = key.split("/", 1)
                if prefix not in CATEGORIES_MAP:
                    continue

                response_class = classify_negative_response(value)
                results.append(
                    {
                        "File": os.path.basename(json_file),
                        "Category": CATEGORIES_MAP[prefix],
                        "Subfield": subfield,
                        "IsIrrelevantSuccess": response_class == "IrrelevantSuccess",
                        "IsNotEnoughOnly": response_class == "NotEnoughOnly",
                        "IsMixedFailure": response_class == "MixedFailure",
                        "IsFailure": response_class in {"MixedFailure", "Failure"},
                        "IsAcceptable": response_class in {"IrrelevantSuccess", "NotEnoughOnly"},
                    }
                )

        except Exception as exc:
            print(f"Error processing {json_file}: {exc}")

    frame = pd.DataFrame(results)
    print(f"Processed {len(frame)} records.")
    return frame


def create_coverage_plot(data_df, condition_col, title_suffix, xlabel, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    subset_df = data_df[data_df[condition_col] == True]
    counts = subset_df.groupby(["Category", "Subfield"]).size().reset_index(name="Count")
    total_files = data_df["File"].nunique()

    category_colors = {
        "Data": "#90C083",
        "Optimisation": "#AEACDD",
        "Model": "#7EB1DD",
        "Evaluation": "#F8AEAE",
    }
    categories = ["Data", "Optimisation", "Model", "Evaluation"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight="bold")
    axes = axes.flatten()

    for index, category in enumerate(categories):
        ax = axes[index]
        subset = counts[counts["Category"] == category]

        all_subfields = data_df[data_df["Category"] == category]["Subfield"].unique()
        full_subset = pd.DataFrame({"Subfield": all_subfields})
        full_subset = full_subset.merge(subset, on="Subfield", how="left").fillna(0)
        full_subset["Subfield"] = full_subset["Subfield"].str.capitalize()
        full_subset = full_subset.sort_values("Count", ascending=True)

        bars = ax.barh(full_subset["Subfield"], full_subset["Count"], color=category_colors[category])
        ax.set_title(category, fontweight="bold", fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 50))

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 10,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                ha="left",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            if width > 0:
                pct = (width / total_files) * 100
                ax.text(
                    width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=14,
                    fontweight="bold",
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {path}")


def create_joint_stacked_plot(data_df, title_suffix, xlabel, filename):
    import matplotlib.patches as mpatches

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    full_counts = (
        data_df[data_df["IsIrrelevantSuccess"] == True]
        .groupby(["Category", "Subfield"])
        .size()
        .reset_index(name="FullCount")
    )
    partial_counts = (
        data_df[data_df["IsNotEnoughOnly"] == True]
        .groupby(["Category", "Subfield"])
        .size()
        .reset_index(name="PartialCount")
    )

    total_files = data_df["File"].nunique()
    categories = ["Data", "Optimisation", "Model", "Evaluation"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight="bold")
    axes = axes.flatten()

    for index, category in enumerate(categories):
        ax = axes[index]
        all_subfields = data_df[data_df["Category"] == category]["Subfield"].unique()
        df_sub = pd.DataFrame({"Subfield": all_subfields})

        f_sub = full_counts[full_counts["Category"] == category]
        p_sub = partial_counts[partial_counts["Category"] == category]

        df_sub = df_sub.merge(f_sub[["Subfield", "FullCount"]], on="Subfield", how="left").fillna(0)
        df_sub = df_sub.merge(p_sub[["Subfield", "PartialCount"]], on="Subfield", how="left").fillna(0)
        df_sub["TotalCount"] = df_sub["FullCount"] + df_sub["PartialCount"]
        df_sub["Subfield"] = df_sub["Subfield"].str.capitalize()
        df_sub = df_sub.sort_values("TotalCount", ascending=True)

        ax.barh(df_sub["Subfield"], df_sub["FullCount"], color="#27AE60", label="Correct rejection")
        ax.barh(
            df_sub["Subfield"],
            df_sub["PartialCount"],
            left=df_sub["FullCount"],
            color="#2980B9",
            label="Only not enough info",
        )

        ax.set_title(category, fontweight="bold", fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 150))

        for row_index, row in df_sub.reset_index(drop=True).iterrows():
            total = row["TotalCount"]
            full = row["FullCount"]
            partial = row["PartialCount"]

            if total > 0:
                ax.text(total + 25, row_index, f"{int(total)}", ha="left", va="center", fontsize=20, fontweight="bold", color="black")
            if full > 0:
                ax.text(full / 2, row_index, f"{int(full)}", ha="center", va="center", color="white", fontsize=18, fontweight="bold")
            if partial > 0:
                ax.text(full + (partial / 2), row_index, f"{int(partial)}", ha="center", va="center", color="white", fontsize=18, fontweight="bold")

    full_patch = mpatches.Patch(color="#27AE60", label="Correct rejection")
    partial_patch = mpatches.Patch(color="#2980B9", label="Only not enough info")
    total_patch = mpatches.Patch(color="black", label="Total (end of bar)")
    fig.legend(
        handles=[full_patch, partial_patch, total_patch],
        loc="upper right",
        fontsize=16,
        bbox_to_anchor=(0.99, 0.99),
        framealpha=0.9,
        edgecolor="black",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=4.0)
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {path}")


def create_joint_grouped_plot(data_df, title_suffix, xlabel, filename):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    full_counts = (
        data_df[data_df["IsIrrelevantSuccess"] == True]
        .groupby(["Category", "Subfield"])
        .size()
        .reset_index(name="FullCount")
    )
    partial_counts = (
        data_df[data_df["IsNotEnoughOnly"] == True]
        .groupby(["Category", "Subfield"])
        .size()
        .reset_index(name="PartialCount")
    )

    total_files = data_df["File"].nunique()
    categories = ["Data", "Optimisation", "Model", "Evaluation"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight="bold")
    axes = axes.flatten()

    for index, category in enumerate(categories):
        ax = axes[index]
        all_subfields = data_df[data_df["Category"] == category]["Subfield"].unique()
        df_sub = pd.DataFrame({"Subfield": all_subfields})

        f_sub = full_counts[full_counts["Category"] == category]
        p_sub = partial_counts[partial_counts["Category"] == category]

        df_sub = df_sub.merge(f_sub[["Subfield", "FullCount"]], on="Subfield", how="left").fillna(0)
        df_sub = df_sub.merge(p_sub[["Subfield", "PartialCount"]], on="Subfield", how="left").fillna(0)
        df_sub["TotalCount"] = df_sub["FullCount"] + df_sub["PartialCount"]
        df_sub["Subfield"] = df_sub["Subfield"].str.capitalize()
        df_sub = df_sub.sort_values("TotalCount", ascending=True)

        y_positions = np.arange(len(df_sub))
        height = 0.35

        bars1 = ax.barh(y_positions - height / 2, df_sub["FullCount"], height, color="#27AE60", label="Correct rejection")
        bars2 = ax.barh(y_positions + height / 2, df_sub["PartialCount"], height, color="#2980B9", label="Only not enough info")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_sub["Subfield"])
        ax.set_title(category, fontweight="bold", fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 50))

        if index == 0:
            ax.legend(fontsize=16)

        for bar in list(bars1) + list(bars2):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 10, bar.get_y() + bar.get_height() / 2, f"{int(width)}", ha="left", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {path}")


def main():
    data_df = load_and_process_data()
    if data_df is None or data_df.empty:
        return

    print("Generating negative-dataset plots...")

    create_coverage_plot(
        data_df,
        "IsIrrelevantSuccess",
        'Correct Negative Abstention: "Not a relevant AI or machine learning publication"',
        "Number of Papers",
        "graph_correct_rejection.png",
    )
    create_coverage_plot(
        data_df,
        "IsNotEnoughOnly",
        'Safe Abstention: Only "Not enough information" Generated',
        "Number of Papers",
        "graph_not_enough_only.png",
    )
    create_coverage_plot(
        data_df,
        "IsMixedFailure",
        "Mixed Failure: Placeholder Plus Extra Generated Content",
        "Number of Papers",
        "graph_mixed_failure.png",
    )
    create_coverage_plot(
        data_df,
        "IsFailure",
        "Failed Negative Abstention: Any Extra Generated Content",
        "Number of Papers",
        "graph_failed_generation.png",
    )
    create_joint_stacked_plot(
        data_df,
        "Acceptable Negative Handling: Correct Rejection & Placeholder-Only",
        "Number of Papers",
        "graph_joint_stacked_success.png",
    )
    create_joint_grouped_plot(
        data_df,
        "Acceptable Negative Handling: Correct Rejection vs Placeholder-Only",
        "Number of Papers",
        "graph_joint_grouped_success.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()