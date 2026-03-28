#!/usr/bin/env python3
"""
generate_categorisation_report.py

Produces two analysis reports explaining how each DOME field is categorised in:

  1. Graph_Panel_V2_Neg/graph_joint_stacked_success.png
     (Negative dataset - Copilot_1012_v2_Neg)
     Categories: Correct Rejection | Not-Enough-Info Rejection | Partial Rejection

  2. Graph_Panel_V2/graph_joint_stacked_yield.png
     (Positive dataset - Copilot_1012_v2_Pos)
     Categories: Full Generation | Partial Generation | Missing | NA

For each field×category combination, up to 10 example PMCIDs with source text
snippets are included so the categorisation can be verified by reading raw data.

Outputs (both saved into Publication_Figures/):
  - categorisation_report_neg.csv
  - categorisation_report_pos.csv
  - categorisation_report_neg.txt   (human-readable doc)
  - categorisation_report_pos.txt   (human-readable doc)
"""

import glob
import json
import os
import re
import string
import textwrap

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NEG_DATA = os.path.join(
    SCRIPT_DIR,
    "../../Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Neg_Processed_2026-03-02_Updated_Metadata",
)
POS_DATA = os.path.join(
    SCRIPT_DIR,
    "../../Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Pos_Processed_2026-03-02",
)
OUT_DIR = SCRIPT_DIR   # reports saved alongside this script in Publication_Figures/

EXAMPLES_PER_CATEGORY = 10
SNIPPET_LENGTH = 400   # characters of raw field value shown in report

CATEGORIES_MAP = {
    "dataset": "Data",
    "optimization": "Optimisation",
    "model": "Model",
    "evaluation": "Evaluation",
}

# ---------------------------------------------------------------------------
# NEGATIVE DATASET CLASSIFICATION  (mirrors generate_graph_panels_negative.py)
# ---------------------------------------------------------------------------
IRRELEVANT_PHRASE = "not a relevant ai or machine learning publication"
MISSING_PATTERN = re.compile(r"not enough information(?: is)?(?: available)?")


def strip_to_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_response_text(value_str: str) -> str:
    text = value_str.lower().replace("\r", "\n").replace("**", "")
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


def classify_negative(value_str) -> str:
    """
    Returns one of:
      IrrelevantSuccess  – only contains "not a relevant ai or ml publication"
      NotEnoughOnly      – only "not enough information" placeholders
      PartialRejection   – blank, or mix of placeholders + real content
      Failure            – real content, none of the rejection signals
    """
    if not isinstance(value_str, str) or not value_str.strip():
        return "PartialRejection"

    normalized = normalize_response_text(value_str)
    if not normalized:
        return "PartialRejection"

    irrelevant_removed = normalized.replace(IRRELEVANT_PHRASE, " ")
    if IRRELEVANT_PHRASE in normalized and not strip_to_alnum(irrelevant_removed):
        return "IrrelevantSuccess"

    allowed_removed = normalized.replace(IRRELEVANT_PHRASE, " ")
    allowed_removed = MISSING_PATTERN.sub(" ", allowed_removed)

    if not strip_to_alnum(allowed_removed):
        if MISSING_PATTERN.search(normalized):
            return "NotEnoughOnly"
        return "PartialRejection"

    if MISSING_PATTERN.search(normalized):
        return "PartialRejection"

    return "Failure"


# ---------------------------------------------------------------------------
# POSITIVE DATASET CLASSIFICATION  (mirrors generate_graph_panels.py)
# ---------------------------------------------------------------------------
def classify_positive(value_str) -> str:
    """
    Returns one of:
      Full     – real content, no missing/NA placeholders remaining
      Partial  – real content AND contains "not enough information" or "not applicable"
      Missing  – only "not enough information" phrases
      NA       – only "not applicable" phrases
      Empty    – nothing after stripping
    """
    if not isinstance(value_str, str) or not value_str.strip():
        return "Empty"

    v_lower = value_str.lower()
    cleaned = re.sub(r"^\s*[\*-]?\s*\*\*?[^*:]+\*\*?\s*:?", "", v_lower, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[\*-]\s+[^*:]+:\s?", "", cleaned, flags=re.MULTILINE)

    for term in [
        "not enough information is available.",
        "not enough information.",
        "not enough information",
        "not applicable.",
        "not applicable",
    ]:
        cleaned = cleaned.replace(term, "")

    chars_left = cleaned.translate(str.maketrans("", "", string.punctuation)).replace(" ", "").replace("\n", "")
    has_valid_text = len(chars_left) > 0
    has_missing_str = "not enough information" in v_lower
    has_na_str = "not applicable" in v_lower

    if not has_valid_text:
        if has_missing_str:
            return "Missing"
        if has_na_str:
            return "NA"
        return "Empty"
    else:
        if has_missing_str or has_na_str:
            return "Partial"
        return "Full"


# ---------------------------------------------------------------------------
# LOAD DATA HELPERS
# ---------------------------------------------------------------------------
def load_neg_records() -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(NEG_DATA, "*.json")):
        pmcid = os.path.basename(path).replace(".json", "")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  [WARN] {path}: {exc}")
            continue
        for key, value in data.items():
            if "/" not in key:
                continue
            prefix, subfield = key.split("/", 1)
            if prefix not in CATEGORIES_MAP:
                continue
            category = classify_negative(value)
            rows.append(
                {
                    "PMCID": pmcid,
                    "Group": CATEGORIES_MAP[prefix],
                    "Subfield": subfield,
                    "Category": category,
                    "RawValue": str(value) if value is not None else "",
                }
            )
    return pd.DataFrame(rows)


def load_pos_records() -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(POS_DATA, "*.json")):
        pmcid = os.path.basename(path).replace(".json", "")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  [WARN] {path}: {exc}")
            continue
        for key, value in data.items():
            if "/" not in key:
                continue
            prefix, subfield = key.split("/", 1)
            if prefix not in CATEGORIES_MAP:
                continue
            category = classify_positive(value)
            rows.append(
                {
                    "PMCID": pmcid,
                    "Group": CATEGORIES_MAP[prefix],
                    "Subfield": subfield,
                    "Category": category,
                    "RawValue": str(value) if value is not None else "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# REPORT BUILDERS
# ---------------------------------------------------------------------------
def build_examples_df(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    For each (Group, Subfield, Category), sample up to EXAMPLES_PER_CATEGORY rows.
    Returns a flat DataFrame ready for CSV export.
    """
    rows = []
    for (group, subfield), g in df.groupby(["Group", "Subfield"]):
        for cat in categories:
            subset = g[g["Category"] == cat]
            sample = subset.head(EXAMPLES_PER_CATEGORY)
            for _, row in sample.iterrows():
                snippet = row["RawValue"].strip()
                rows.append(
                    {
                        "Group": group,
                        "Subfield": subfield,
                        "DOME_Field": f"{group}/{subfield}",
                        "PMCID": row["PMCID"],
                        "Category": cat,
                        "Source_Snippet": snippet,
                    }
                )
    return pd.DataFrame(rows)


def write_text_report(df_examples: pd.DataFrame,
                      categories: list,
                      dataset_name: str,
                      methodology_section: str,
                      out_path: str):
    """Write a human-readable .txt report."""
    lines = []
    sep = "=" * 100

    lines.append(sep)
    lines.append(f"CATEGORISATION REPORT — {dataset_name}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(sep)
    lines.append("")
    lines.append("SECTION 1 — METHODOLOGY")
    lines.append("-" * 80)
    lines += textwrap.wrap(methodology_section, width=100)
    lines.append("")

    lines.append("SECTION 2 — EXAMPLES BY FIELD AND CATEGORY")
    lines.append("-" * 80)
    lines.append(
        "For each DOME field × category combination, up to "
        f"{EXAMPLES_PER_CATEGORY} examples are shown. "
        "Each entry includes: PMCID | Category | Full source text of the field value."
    )
    lines.append("")

    for (group, subfield), g_ex in df_examples.groupby(["Group", "Subfield"]):
        field_label = f"{group}/{subfield}"
        lines.append(sep)
        lines.append(f"FIELD: {field_label}")
        lines.append(sep)

        # Overall counts for this field
        total_df = None  # not passed; counts from examples df only approximate
        for cat in categories:
            cat_examples = g_ex[g_ex["Category"] == cat]
            count = len(cat_examples)
            if count == 0:
                lines.append(f"  [{cat}]  — No examples found")
                lines.append("")
                continue
            lines.append(f"  [{cat}]  — {count} example(s) shown below")
            lines.append("")
            for i, (_, row) in enumerate(cat_examples.iterrows(), 1):
                lines.append(f"    Example {i}")
                lines.append(f"      PMCID   : {row['PMCID']}")
                lines.append(f"      Category: {row['Category']}")
                raw_lines = row["Source_Snippet"].splitlines()
                display_lines = []
                for raw_line in raw_lines:
                    if raw_line.strip():
                        wrapped = textwrap.wrap(raw_line, width=90)
                        display_lines.extend(wrapped if wrapped else [raw_line])
                    else:
                        display_lines.append("")
                if display_lines:
                    lines.append(f"      Snippet : {display_lines[0]}")
                    for cont in display_lines[1:]:
                        lines.append(f"                {cont}")
                else:
                    lines.append(f"      Snippet : (empty)")
                lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  [OK] Text report -> {out_path}")


# ---------------------------------------------------------------------------
# METHODOLOGY STRINGS
# ---------------------------------------------------------------------------
NEG_METHODOLOGY = """
GRAPH: Graph_Panel_V2_Neg/graph_joint_stacked_success.png
SOURCE DATA: Copilot_1012_v2_Neg_Processed_2026-03-02_Updated_Metadata  (negative dataset — papers that are
NOT relevant ML publications; Copilot should ideally produce rejection / placeholder responses only).

CLASSIFICATION LOGIC (from generate_graph_panels_negative.py):

Each JSON field value goes through a two-step text normalisation before classification:
  Step 1: Lowercase, remove markdown bold (**), strip leading bullet/dash characters, collapse whitespace.
  Step 2: Strip structural sub-headers (anything matching "Short Label: ") iteratively until stable.

The normalised text is then matched against three ordered rules:

  (A) IrrelevantSuccess  ("Correct Rejection" — red in graph)
      The entire normalised response, after removing the phrase "not a relevant ai or machine
      learning publication", contains ONLY whitespace / punctuation.  The Copilot correctly
      identified the paper as irrelevant and produced nothing else.

  (B) NotEnoughOnly  ("Not-Enough-Info Rejection" — light blue in graph)
      After removing both IRRELEVANT_PHRASE and all "not enough information [is] [available]"
      matches, NOTHING alphanumeric remains.  At least one "not enough information" match must
      be present.  The Copilot only produced placeholder phrases — no real content.

  (C) PartialRejection  ("Partial Rejection" — orange in graph)
      The response is blank/None, OR the cleaned text still contains alphanumeric content
      alongside "not enough information" placeholders, OR none of the other patterns match.
      The Copilot produced some real content when it should have produced none.

  (D) Failure  (NOT plotted in the success graph)
      Real content present with no rejection/placeholder signals at all.

For the stacked bar graph only categories A, B, and C are plotted (all represent some form
of acceptable or semi-acceptable Copilot rejection behaviour for negative papers).
""".strip()

POS_METHODOLOGY = """
GRAPH: Graph_Panel_V2/graph_joint_stacked_yield.png
SOURCE DATA: Copilot_1012_v2_Pos_Processed_2026-03-02  (positive dataset — papers that ARE
relevant ML publications; Copilot should produce real DOME field content).

CLASSIFICATION LOGIC (from generate_graph_panels.py  ->  determine_yield_category):

Each field value is processed as follows:
  Step 1: Lowercase the value.
  Step 2: Strip markdown bold/italic headers using regex (lines starting with **/- followed by
          a label and colon).
  Step 3: Remove standard placeholder strings:
            "not enough information is available."
            "not enough information."
            "not enough information"
            "not applicable."
            "not applicable"
  Step 4: Strip all punctuation and whitespace from what remains.
  Step 5: Check whether any alphanumeric characters are left (has_valid_text).

Decision tree:

  has_valid_text = False
    → has "not enough information" in original? → Missing
    → has "not applicable" in original?         → NA
    → otherwise                                  → Empty

  has_valid_text = True
    → original also contains "not enough information" OR "not applicable"? → Partial
    → otherwise                                                              → Full

For the stacked yield graph:
  Full    (green)   — complete, valid DOME field extraction with no missing indicators
  Partial (blue)    — some valid content but at least one sub-question answered as
                      "not enough information" or "not applicable"
  Missing / NA / Empty are NOT shown in the stacked yield graph (they appear in separate panels).
""".strip()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Loading NEGATIVE dataset records...")
    df_neg = load_neg_records()
    print(f"  -> {len(df_neg)} records from {df_neg['PMCID'].nunique()} files")

    neg_categories = ["IrrelevantSuccess", "NotEnoughOnly", "PartialRejection", "Failure"]

    print("\nBuilding negative examples table...")
    df_neg_examples = build_examples_df(df_neg, neg_categories)

    neg_csv = os.path.join(OUT_DIR, "categorisation_report_neg.csv")
    df_neg_examples.to_csv(neg_csv, index=False)
    print(f"  [OK] CSV -> {neg_csv}")

    neg_txt = os.path.join(OUT_DIR, "categorisation_report_neg.txt")
    write_text_report(
        df_neg_examples,
        neg_categories,
        "NEGATIVE DATASET — graph_joint_stacked_success.png",
        NEG_METHODOLOGY,
        neg_txt,
    )

    # Summary counts for neg
    neg_summary = (
        df_neg.groupby(["Group", "Subfield", "Category"])
        .size()
        .reset_index(name="Count")
    )
    neg_summary_csv = os.path.join(OUT_DIR, "categorisation_summary_neg.csv")
    neg_summary.to_csv(neg_summary_csv, index=False)
    print(f"  [OK] Summary counts CSV -> {neg_summary_csv}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Loading POSITIVE dataset records...")
    df_pos = load_pos_records()
    print(f"  -> {len(df_pos)} records from {df_pos['PMCID'].nunique()} files")

    pos_categories = ["Full", "Partial", "Missing", "NA", "Empty"]

    print("\nBuilding positive examples table...")
    df_pos_examples = build_examples_df(df_pos, pos_categories)

    pos_csv = os.path.join(OUT_DIR, "categorisation_report_pos.csv")
    df_pos_examples.to_csv(pos_csv, index=False)
    print(f"  [OK] CSV -> {pos_csv}")

    pos_txt = os.path.join(OUT_DIR, "categorisation_report_pos.txt")
    write_text_report(
        df_pos_examples,
        pos_categories,
        "POSITIVE DATASET — graph_joint_stacked_yield.png",
        POS_METHODOLOGY,
        pos_txt,
    )

    # Summary counts for pos
    pos_summary = (
        df_pos.groupby(["Group", "Subfield", "Category"])
        .size()
        .reset_index(name="Count")
    )
    pos_summary_csv = os.path.join(OUT_DIR, "categorisation_summary_pos.csv")
    pos_summary.to_csv(pos_summary_csv, index=False)
    print(f"  [OK] Summary counts CSV -> {pos_summary_csv}")

    print("\n" + "=" * 60)
    print("All reports written to:", OUT_DIR)
    print("Files created:")
    for f in [neg_csv, neg_txt, neg_summary_csv, pos_csv, pos_txt, pos_summary_csv]:
        size_kb = os.path.getsize(f) / 1024
        print(f"  {os.path.basename(f):45s}  {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
