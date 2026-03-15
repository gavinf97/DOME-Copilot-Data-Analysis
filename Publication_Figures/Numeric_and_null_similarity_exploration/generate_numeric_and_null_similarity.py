#!/usr/bin/env python3
"""
generate_numeric_and_null_similarity.py

Deconflated similarity analysis between:
  - Copilot 222 v2 processed JSON fields
  - DOME Registry Human Reviews 258 JSON entries

Mapped by DOI, excluding publication/* fields.

Produces three separate analyses/graphs:
1) Numeric/percentage direct matching (where either side contains numeric info)
2) "Not enough information" alignment vs registry strong-negative/null responses
3) Yes/No style response alignment
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

COPILOT_DIR = os.path.join(
    WORKSPACE_ROOT,
    "Copilot_Processed_Datasets_JSON",
    "Copilot_222_v2_Processed_2026-03-02_Updated_Metadata",
)

HUMAN_FILE = os.path.join(
    WORKSPACE_ROOT,
    "DOME_Registry_Human_Reviews_258_20260205.json",
)

OUTPUT_DIR = SCRIPT_DIR

GRAPH_FILENAMES = [
    "graph_1_numeric_percent_direct_match_by_field.png",
    "graph_1b_numeric_percent_breakdown_by_field.png",
    "graph_2_not_enough_info_alignment_by_field.png",
    "graph_3_yesno_alignment_by_field.png",
    "graph_3b_yesno_breakdown_by_field.png",
    "graph_4_url_alignment_by_field.png",
    "graph_4b_url_breakdown_by_field.png",
]
GRAPH_DATA_FILENAME = "numeric_null_similarity_graph_data.csv"

MISSING_REGEX = re.compile(r"not enough information(?: is)?(?: available)?", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://[^\s\]\[\)\(\"'<>]+", re.IGNORECASE)
PERCENT_REGEX = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%")
YES_REGEX = re.compile(r"\byes\b", re.IGNORECASE)
NO_REGEX = re.compile(r"\bno\b", re.IGNORECASE)

NULL_LIKE_VALUES = {
    "",
    "no",
    "none",
    "na",
    "n/a",
    "not applicable",
    "not applicable.",
    "not enough information",
    "not enough information.",
    "not enough information is available",
    "not enough information is available.",
    "unknown",
    "null",
}

STRONG_NEGATIVE_PHRASES = {
    "no",
    "none",
    "not applicable",
    "not applicable.",
    "not available",
    "not available.",
    "not reported",
    "not reported.",
    "not provided",
    "not provided.",
    "unknown",
    "na",
    "n/a",
    "unavailable",
    "not done",
}


def normalize_doi(doi):
    if doi is None:
        return ""
    return str(doi).strip().lower()


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_null_like(value):
    text = normalize_text(value)
    if text in NULL_LIKE_VALUES:
        return True

    text_stripped = text.rstrip(".;:,")
    if text_stripped in NULL_LIKE_VALUES:
        return True

    return False


def is_strong_negative(value):
    text = normalize_text(value)
    if text in STRONG_NEGATIVE_PHRASES:
        return True

    if re.search(r"\b(no|none|not applicable|not available|not reported|not provided|unknown|unavailable)\b", text):
        return True

    return False


def extract_answer_segments(value):
    """
    Extract likely answer segments, accounting for sub-question formatting where
    answers often appear after ':'.
    """
    if value is None:
        return []

    text = str(value)
    segments = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            after = line.split(":", 1)[1].strip()
            segments.append(after if after else line)
        else:
            segments.append(line)

    return segments if segments else [text]


def copilot_missing_mode(value):
    """
    Returns:
      - "missing_only": contains missing phrase and no other substantive content
      - "partial_missing": contains missing phrase plus additional content
      - "other": all other cases
    """
    if value is None:
        return "other"

    text = str(value)
    if MISSING_REGEX.search(text) is None:
        return "other"

    segments = extract_answer_segments(value)
    has_nei_segment = False
    has_partial_segment = False
    has_other_content = False

    for seg in segments:
        seg_lower = seg.lower()
        if "not enough information" in seg_lower:
            has_nei_segment = True
            cleaned = MISSING_REGEX.sub("", seg_lower)
            cleaned = re.sub(r"[\s\.\,\:\;\-\_\*\|\(\)\[\]\{\}]+", "", cleaned)
            if cleaned:
                has_partial_segment = True
        else:
            compact = re.sub(r"[\s\.\,\:\;\-\_\*\|\(\)\[\]\{\}]+", "", seg_lower)
            if compact:
                has_other_content = True

    if has_nei_segment and not has_partial_segment and not has_other_content:
        return "missing_only"

    if has_nei_segment:
        return "partial_missing"

    return "other"


def extract_numbers(value):
    if value is None:
        return []

    text = str(value)
    raw = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    nums = []
    for token in raw:
        try:
            nums.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return nums


def extract_percentages(value):
    if value is None:
        return []

    text = str(value)
    raw = PERCENT_REGEX.findall(text)
    values = []
    for token in raw:
        cleaned = token.replace("%", "").replace(",", "").strip()
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def has_any_numeric_or_percent(value):
    return bool(extract_numbers(value) or extract_percentages(value))


def extract_urls(value):
    if value is None:
        return set()

    text = str(value)
    urls = set()
    for raw in URL_REGEX.findall(text):
        normalized = raw.strip().rstrip(".,;:!?)\"]}").lower()
        if normalized:
            urls.add(normalized)
    return urls


def score_url_overlap(copilot_value, human_value):
    """
    URL scoring:
      - 1.0 when URL sets are exactly the same and non-empty
      - 0.5 when there is partial overlap (intersection non-empty, sets differ,
        and at least one side has multiple URLs)
      - 0.0 otherwise
    """
    copilot_urls = extract_urls(copilot_value)
    human_urls = extract_urls(human_value)

    if not copilot_urls or not human_urls:
        return 0.0, "no_url_match"

    if copilot_urls == human_urls:
        return 1.0, "url_full_match"

    overlap = copilot_urls.intersection(human_urls)
    if overlap and (len(copilot_urls) > 1 or len(human_urls) > 1):
        return 0.5, "url_partial_match"

    return 0.0, "no_url_match"


def has_numeric_match(copilot_value, human_value):
    c_nums = extract_numbers(copilot_value)
    h_nums = extract_numbers(human_value)

    if not c_nums or not h_nums:
        return False

    for c_num in c_nums:
        for h_num in h_nums:
            if np.isclose(c_num, h_num, rtol=1e-4, atol=1e-8):
                return True
    return False


def has_percent_match(copilot_value, human_value):
    c_perc = extract_percentages(copilot_value)
    h_perc = extract_percentages(human_value)

    if not c_perc or not h_perc:
        return False

    for c_num in c_perc:
        for h_num in h_perc:
            if np.isclose(c_num, h_num, rtol=1e-4, atol=1e-8):
                return True
    return False


def classify_yes_no(value):
    """
    Classify response into yes/no/mixed/none.
    Uses answer segments so copilot sub-question text before ':' does not dominate.
    """
    segments = extract_answer_segments(value)
    has_yes = False
    has_no = False

    for seg in segments:
        if YES_REGEX.search(seg):
            has_yes = True
        if NO_REGEX.search(seg):
            has_no = True

    if has_yes and has_no:
        return "mixed"
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return "none"


def load_human_registry():
    with open(HUMAN_FILE, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    doi_map = {}
    for entry in records:
        doi = normalize_doi(entry.get("publication", {}).get("doi", ""))
        if doi:
            doi_map[doi] = entry
    return doi_map


def get_human_field(human_entry, field_key):
    """
    Copilot key style: category/subfield
    Human style: human_entry[category][subfield]
    """
    if "/" not in field_key:
        return ""

    category, subfield = field_key.split("/", 1)

    if category == "publication":
        return ""

    cat_obj = human_entry.get(category, {})
    if not isinstance(cat_obj, dict):
        return ""

    return cat_obj.get(subfield, "")


def finalize_summary_plot(plot_df, title, out_path):
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("SuccessPercent", ascending=True)
    plt.figure(figsize=(14, max(8, 0.35 * len(plot_df))))
    y_vals = np.arange(len(plot_df))
    bars = plt.barh(
        y_vals,
        plot_df["SuccessPercent"],
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.6,
    )

    plt.yticks(y_vals, plot_df["Field"], fontsize=9)
    plt.xlabel("Match Success (%)", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlim(0, 100)

    x_pad = 1.2
    for bar, compared_count in zip(bars, plot_df["ComparedCount"]):
        x_val = bar.get_width()
        y_val = bar.get_y() + bar.get_height() / 2
        plt.text(
            x_val + x_pad,
            y_val,
            f"n={int(compared_count)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def finalize_stacked_breakdown_plot(plot_df, title, out_path, components):
    """
    Plot horizontal stacked percentage bars.

    components: list of tuples (column_name, legend_label, color_hex)
    Expects TotalCount and ComparableCount columns in plot_df.
    """
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("ComparableCount", ascending=True)
    plt.figure(figsize=(15, max(8, 0.35 * len(plot_df))))
    y_vals = np.arange(len(plot_df))

    left = np.zeros(len(plot_df))
    for col_name, label, color in components:
        vals = plot_df[col_name].to_numpy()
        plt.barh(
            y_vals,
            vals,
            left=left,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        left = left + vals

    plt.yticks(y_vals, plot_df["Field"], fontsize=9)
    plt.xlabel("Breakdown (%) within relevant rows", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlim(0, 100)

    for i, row in plot_df.reset_index(drop=True).iterrows():
        y_val = y_vals[i]
        plt.text(
            100.5,
            y_val,
            f"n_cmp={int(row['ComparableCount'])}, n_rel={int(row['TotalCount'])}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.legend(loc="lower right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def remove_previous_outputs():
    """
    Keep this folder tidy by removing prior generated CSV/JSON files and PNGs
    that are not part of the current graph output set.
    """
    keep_png = set(GRAPH_FILENAMES)
    keep_data = {GRAPH_DATA_FILENAME}

    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.isfile(path):
            continue

        lower = name.lower()
        if lower.endswith(".csv") and name not in keep_data:
            os.remove(path)
        elif lower.endswith(".json"):
            os.remove(path)
        elif lower.endswith(".png") and name not in keep_png:
            os.remove(path)


def build_consolidated_graph_data(
    numeric_field_summary,
    nei_field_summary,
    yesno_field_summary,
    url_field_summary,
    numeric_breakdown_df,
    yesno_breakdown_df,
    url_breakdown_df,
):
    tables = [
        ("graph_1_numeric_percent_direct_match", numeric_field_summary),
        ("graph_2_not_enough_info_alignment", nei_field_summary),
        ("graph_3_yesno_alignment", yesno_field_summary),
        ("graph_4_url_alignment", url_field_summary),
        ("graph_1b_numeric_percent_breakdown", numeric_breakdown_df),
        ("graph_3b_yesno_breakdown", yesno_breakdown_df),
        ("graph_4b_url_breakdown", url_breakdown_df),
    ]

    merged = []
    for table_name, df in tables:
        if df is None or df.empty:
            continue
        out = df.copy()
        out.insert(0, "GraphTable", table_name)
        merged.append(out)

    if not merged:
        return pd.DataFrame(columns=["GraphTable"])

    return pd.concat(merged, ignore_index=True, sort=False)


def run_analysis():
    if not os.path.exists(COPILOT_DIR):
        raise FileNotFoundError(f"Copilot directory not found: {COPILOT_DIR}")
    if not os.path.exists(HUMAN_FILE):
        raise FileNotFoundError(f"Human registry file not found: {HUMAN_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    remove_previous_outputs()

    human_by_doi = load_human_registry()
    copilot_files = sorted(glob.glob(os.path.join(COPILOT_DIR, "*.json")))
    if not copilot_files:
        raise FileNotFoundError(f"No JSON files found in: {COPILOT_DIR}")

    rows = []
    matched_docs = 0
    unmatched_docs = 0

    for file_path in copilot_files:
        with open(file_path, "r", encoding="utf-8") as handle:
            copilot_data = json.load(handle)

        doi = normalize_doi(copilot_data.get("publication/doi", ""))
        if not doi or doi not in human_by_doi:
            unmatched_docs += 1
            continue

        matched_docs += 1
        human_entry = human_by_doi[doi]
        pmcid = copilot_data.get("publication/pmcid", os.path.basename(file_path).replace(".json", ""))

        for key, copilot_value in copilot_data.items():
            if not isinstance(key, str):
                continue
            if "/" not in key:
                continue
            if key.startswith("publication/"):
                continue

            human_value = get_human_field(human_entry, key)
            copilot_has_numericish = has_any_numeric_or_percent(copilot_value)
            human_has_numericish = has_any_numeric_or_percent(human_value)
            numeric_or_percent_presence = copilot_has_numericish or human_has_numericish
            numeric_or_percent_comparable = copilot_has_numericish and human_has_numericish

            percent_match = has_percent_match(copilot_value, human_value)
            numeric_match = has_numeric_match(copilot_value, human_value)
            direct_numeric_percent_match = percent_match or numeric_match

            nei_mode = copilot_missing_mode(copilot_value)
            human_negative = is_null_like(human_value) or is_strong_negative(human_value)
            nei_relevant = nei_mode in {"missing_only", "partial_missing"}
            if nei_mode == "missing_only" and human_negative:
                nei_score = 1.0
                nei_rule = "missing_to_negative_match"
            elif nei_mode == "partial_missing" and human_negative:
                nei_score = 0.5
                nei_rule = "partial_missing_to_negative_half_match"
            else:
                nei_score = 0.0
                nei_rule = "no_nei_match"

            yesno_copilot = classify_yes_no(copilot_value)
            yesno_human = classify_yes_no(human_value)
            yesno_relevant = yesno_copilot != "none" or yesno_human != "none"
            yesno_comparable = yesno_copilot != "none" and yesno_human != "none"
            if yesno_comparable:
                if yesno_copilot == yesno_human:
                    yesno_score = 1.0
                    yesno_rule = "yesno_match"
                elif "mixed" in {yesno_copilot, yesno_human}:
                    yesno_score = 0.5
                    yesno_rule = "yesno_partial_mixed"
                else:
                    yesno_score = 0.0
                    yesno_rule = "yesno_mismatch"
            else:
                yesno_score = 0.0
                yesno_rule = "yesno_not_comparable"

            copilot_urls = extract_urls(copilot_value)
            human_urls = extract_urls(human_value)
            url_relevant = bool(copilot_urls or human_urls)
            url_comparable = bool(copilot_urls and human_urls)
            url_score, url_rule = score_url_overlap(copilot_value, human_value)

            category = key.split("/", 1)[0]
            rows.append(
                {
                    "PMCID": pmcid,
                    "DOI": doi,
                    "Field": key,
                    "Category": category,
                    "CopilotValue": "" if copilot_value is None else str(copilot_value),
                    "HumanValue": "" if human_value is None else str(human_value),
                    "CopilotHasNumericOrPercent": copilot_has_numericish,
                    "HumanHasNumericOrPercent": human_has_numericish,
                    "NumericOrPercentPresence": numeric_or_percent_presence,
                    "NumericOrPercentComparable": numeric_or_percent_comparable,
                    "NumericMatch": numeric_match,
                    "PercentMatch": percent_match,
                    "DirectNumericPercentMatch": direct_numeric_percent_match,
                    "CopilotMissingMode": nei_mode,
                    "HumanStrongNegative": human_negative,
                    "NeiRelevant": nei_relevant,
                    "NeiScore": nei_score,
                    "NeiRule": nei_rule,
                    "CopilotYesNo": yesno_copilot,
                    "HumanYesNo": yesno_human,
                    "YesNoRelevant": yesno_relevant,
                    "YesNoComparable": yesno_comparable,
                    "YesNoScore": yesno_score,
                    "YesNoRule": yesno_rule,
                    "CopilotUrlCount": len(copilot_urls),
                    "HumanUrlCount": len(human_urls),
                    "UrlRelevant": url_relevant,
                    "UrlComparable": url_comparable,
                    "UrlScore": url_score,
                    "UrlRule": url_rule,
                }
            )

    if not rows:
        raise RuntimeError("No matched DOI records produced field comparisons.")

    frame = pd.DataFrame(rows)

    # Analysis 1: Numeric/Percent direct matching (strictly comparable rows only).
    numeric_relevant = frame[frame["NumericOrPercentComparable"]].copy()
    numeric_field_summary = (
        numeric_relevant.groupby(["Category", "Field"], as_index=False)
        .agg(
            ComparedCount=("DirectNumericPercentMatch", "count"),
            MatchCount=("DirectNumericPercentMatch", lambda series: int(series.sum())),
        )
    )
    if not numeric_field_summary.empty:
        numeric_field_summary["SuccessPercent"] = (
            numeric_field_summary["MatchCount"] / numeric_field_summary["ComparedCount"]
        ) * 100.0

    # Analysis 2: NEI alignment against strong-negative/null human responses.
    nei_relevant = frame[frame["NeiRelevant"]].copy()
    nei_field_summary = (
        nei_relevant.groupby(["Category", "Field"], as_index=False)
        .agg(
            ComparedCount=("NeiScore", "count"),
            MeanScore=("NeiScore", "mean"),
            FullMatches=("NeiScore", lambda series: int((series == 1.0).sum())),
            HalfMatches=("NeiScore", lambda series: int((series == 0.5).sum())),
        )
    )
    if not nei_field_summary.empty:
        nei_field_summary["SuccessPercent"] = nei_field_summary["MeanScore"] * 100.0

    # Analysis 3: Yes/No style alignment (strictly comparable rows only).
    yesno_relevant = frame[frame["YesNoComparable"]].copy()
    yesno_field_summary = (
        yesno_relevant.groupby(["Category", "Field"], as_index=False)
        .agg(
            ComparedCount=("YesNoScore", "count"),
            MeanScore=("YesNoScore", "mean"),
            FullMatches=("YesNoScore", lambda series: int((series == 1.0).sum())),
            HalfMatches=("YesNoScore", lambda series: int((series == 0.5).sum())),
        )
    )
    if not yesno_field_summary.empty:
        yesno_field_summary["SuccessPercent"] = yesno_field_summary["MeanScore"] * 100.0

    # Analysis 4: URL matching (strictly comparable rows only).
    url_relevant = frame[frame["UrlComparable"]].copy()
    url_field_summary = (
        url_relevant.groupby(["Category", "Field"], as_index=False)
        .agg(
            ComparedCount=("UrlScore", "count"),
            MeanScore=("UrlScore", "mean"),
            FullMatches=("UrlScore", lambda series: int((series == 1.0).sum())),
            HalfMatches=("UrlScore", lambda series: int((series == 0.5).sum())),
        )
    )
    if not url_field_summary.empty:
        url_field_summary["SuccessPercent"] = url_field_summary["MeanScore"] * 100.0

    # Detailed stacked breakdowns for source/mismatch clarity.
    numeric_breakdown_rows = []
    for (category, field), group in frame[frame["NumericOrPercentPresence"]].groupby(["Category", "Field"]):
        total = len(group)
        comparable = int(group["NumericOrPercentComparable"].sum())
        match = int((group["NumericOrPercentComparable"] & group["DirectNumericPercentMatch"]).sum())
        both_no_match = int((group["NumericOrPercentComparable"] & (~group["DirectNumericPercentMatch"])).sum())
        copilot_only = int((group["CopilotHasNumericOrPercent"] & (~group["HumanHasNumericOrPercent"])).sum())
        human_only = int((group["HumanHasNumericOrPercent"] & (~group["CopilotHasNumericOrPercent"])).sum())

        numeric_breakdown_rows.append(
            {
                "Category": category,
                "Field": field,
                "TotalCount": total,
                "ComparableCount": comparable,
                "PctMatch": (match / total) * 100.0 if total else 0.0,
                "PctBothPresentNoMatch": (both_no_match / total) * 100.0 if total else 0.0,
                "PctCopilotOnly": (copilot_only / total) * 100.0 if total else 0.0,
                "PctHumanOnly": (human_only / total) * 100.0 if total else 0.0,
            }
        )

    yesno_breakdown_rows = []
    yesno_scope = frame[frame["YesNoRelevant"]]
    for (category, field), group in yesno_scope.groupby(["Category", "Field"]):
        total = len(group)
        comparable = int(group["YesNoComparable"].sum())
        full_match = int((group["YesNoComparable"] & (group["YesNoScore"] == 1.0)).sum())
        partial_match = int((group["YesNoComparable"] & (group["YesNoScore"] == 0.5)).sum())
        both_no_match = int((group["YesNoComparable"] & (group["YesNoScore"] == 0.0)).sum())
        copilot_only = int(((group["CopilotYesNo"] != "none") & (group["HumanYesNo"] == "none")).sum())
        human_only = int(((group["HumanYesNo"] != "none") & (group["CopilotYesNo"] == "none")).sum())

        yesno_breakdown_rows.append(
            {
                "Category": category,
                "Field": field,
                "TotalCount": total,
                "ComparableCount": comparable,
                "PctFullMatch": (full_match / total) * 100.0 if total else 0.0,
                "PctPartialMixed": (partial_match / total) * 100.0 if total else 0.0,
                "PctBothPresentNoMatch": (both_no_match / total) * 100.0 if total else 0.0,
                "PctCopilotOnly": (copilot_only / total) * 100.0 if total else 0.0,
                "PctHumanOnly": (human_only / total) * 100.0 if total else 0.0,
            }
        )

    url_breakdown_rows = []
    url_scope = frame[frame["UrlRelevant"]]
    for (category, field), group in url_scope.groupby(["Category", "Field"]):
        total = len(group)
        comparable = int(group["UrlComparable"].sum())
        full_match = int((group["UrlComparable"] & (group["UrlScore"] == 1.0)).sum())
        partial_match = int((group["UrlComparable"] & (group["UrlScore"] == 0.5)).sum())
        both_no_match = int((group["UrlComparable"] & (group["UrlScore"] == 0.0)).sum())
        copilot_only = int(((group["CopilotUrlCount"] > 0) & (group["HumanUrlCount"] == 0)).sum())
        human_only = int(((group["HumanUrlCount"] > 0) & (group["CopilotUrlCount"] == 0)).sum())

        url_breakdown_rows.append(
            {
                "Category": category,
                "Field": field,
                "TotalCount": total,
                "ComparableCount": comparable,
                "PctFullMatch": (full_match / total) * 100.0 if total else 0.0,
                "PctPartialMatch": (partial_match / total) * 100.0 if total else 0.0,
                "PctBothPresentNoMatch": (both_no_match / total) * 100.0 if total else 0.0,
                "PctCopilotOnlyUrl": (copilot_only / total) * 100.0 if total else 0.0,
                "PctHumanOnlyUrl": (human_only / total) * 100.0 if total else 0.0,
            }
        )

    numeric_breakdown_df = pd.DataFrame(numeric_breakdown_rows)
    yesno_breakdown_df = pd.DataFrame(yesno_breakdown_rows)
    url_breakdown_df = pd.DataFrame(url_breakdown_rows)

    graph_data = build_consolidated_graph_data(
        numeric_field_summary,
        nei_field_summary,
        yesno_field_summary,
        url_field_summary,
        numeric_breakdown_df,
        yesno_breakdown_df,
        url_breakdown_df,
    )
    graph_data.to_csv(os.path.join(OUTPUT_DIR, GRAPH_DATA_FILENAME), index=False)

    # Create three deconflated field-level graphs with n labels at each bar end.
    finalize_summary_plot(
        numeric_field_summary,
        "Direct Numeric/Percentage Match by Field (n shown at bar end)",
        os.path.join(OUTPUT_DIR, "graph_1_numeric_percent_direct_match_by_field.png"),
    )
    finalize_summary_plot(
        nei_field_summary,
        '"Not enough information" Alignment vs Registry Negative/Null by Field (n shown at bar end)',
        os.path.join(OUTPUT_DIR, "graph_2_not_enough_info_alignment_by_field.png"),
    )
    finalize_summary_plot(
        yesno_field_summary,
        "Yes/No Alignment by Field (n shown at bar end)",
        os.path.join(OUTPUT_DIR, "graph_3_yesno_alignment_by_field.png"),
    )
    finalize_summary_plot(
        url_field_summary,
        "URL Alignment by Field (where any URL present; n shown at bar end)",
        os.path.join(OUTPUT_DIR, "graph_4_url_alignment_by_field.png"),
    )

    # Stacked breakdown graphs to show source of non-match and one-sided availability.
    finalize_stacked_breakdown_plot(
        numeric_breakdown_df,
        "Numeric/Percentage Breakdown by Field (source-aware)",
        os.path.join(OUTPUT_DIR, "graph_1b_numeric_percent_breakdown_by_field.png"),
        components=[
            ("PctMatch", "Match", "#27AE60"),
            ("PctBothPresentNoMatch", "Both present, no match", "#E67E22"),
            ("PctCopilotOnly", "Copilot only numeric/percent", "#3498DB"),
            ("PctHumanOnly", "Registry only numeric/percent", "#8E44AD"),
        ],
    )
    finalize_stacked_breakdown_plot(
        yesno_breakdown_df,
        "Yes/No Breakdown by Field (source-aware)",
        os.path.join(OUTPUT_DIR, "graph_3b_yesno_breakdown_by_field.png"),
        components=[
            ("PctFullMatch", "Full yes/no match", "#27AE60"),
            ("PctPartialMixed", "Partial mixed", "#F1C40F"),
            ("PctBothPresentNoMatch", "Both present, mismatch", "#E67E22"),
            ("PctCopilotOnly", "Copilot only yes/no", "#3498DB"),
            ("PctHumanOnly", "Registry only yes/no", "#8E44AD"),
        ],
    )
    finalize_stacked_breakdown_plot(
        url_breakdown_df,
        "URL Breakdown by Field (who had URL when not matching)",
        os.path.join(OUTPUT_DIR, "graph_4b_url_breakdown_by_field.png"),
        components=[
            ("PctFullMatch", "Both had URL and fully matched", "#27AE60"),
            ("PctPartialMatch", "Both had URL and partially matched", "#F1C40F"),
            ("PctBothPresentNoMatch", "Both had URL but no match", "#E67E22"),
            ("PctCopilotOnlyUrl", "URL only in Copilot", "#3498DB"),
            ("PctHumanOnlyUrl", "URL only in Registry 258", "#8E44AD"),
        ],
    )

    summary = {
        "copilot_json_files_total": len(copilot_files),
        "doi_matched_documents": matched_docs,
        "doi_unmatched_documents": unmatched_docs,
        "field_comparisons_total": int(len(frame)),
        "analysis_1_numeric_percent_relevant_rows": int(len(frame[frame["NumericOrPercentPresence"]])),
        "analysis_1_numeric_percent_comparable_rows": int(len(numeric_relevant)),
        "analysis_1_overall_success_percent": float(
            numeric_relevant["DirectNumericPercentMatch"].mean() * 100.0
        ) if not numeric_relevant.empty else 0.0,
        "analysis_2_nei_relevant_rows": int(len(nei_relevant)),
        "analysis_2_overall_success_percent": float(
            nei_relevant["NeiScore"].mean() * 100.0
        ) if not nei_relevant.empty else 0.0,
        "analysis_3_yesno_relevant_rows": int(len(frame[frame["YesNoRelevant"]])),
        "analysis_3_yesno_comparable_rows": int(len(yesno_relevant)),
        "analysis_3_overall_success_percent": float(
            yesno_relevant["YesNoScore"].mean() * 100.0
        ) if not yesno_relevant.empty else 0.0,
        "analysis_4_url_relevant_rows": int(len(frame[frame["UrlRelevant"]])),
        "analysis_4_url_comparable_rows": int(len(url_relevant)),
        "analysis_4_overall_success_percent": float(
            url_relevant["UrlScore"].mean() * 100.0
        ) if not url_relevant.empty else 0.0,
    }

    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Graph data file: {os.path.join(OUTPUT_DIR, GRAPH_DATA_FILENAME)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_analysis()
