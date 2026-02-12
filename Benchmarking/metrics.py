import os
import json
import pandas as pd
import nltk
import evaluate
from bert_score import score as bert_score


def ensure_nltk_data():
    resources = [
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

# call this at the top of script
ensure_nltk_data()

# -------------------------------------------------------------------
# 0. Load DOME registry 
# -------------------------------------------------------------------
print("0. Load DOME registry ")

registry_file_path = 'results/DOME_Registry_CSV_Files/PMCIDs_DOME_Registry_Contents_2025-11-25_merged.tsv'

# Load registry CSV directly into DataFrame
df_reg = pd.read_csv(registry_file_path, sep='\t')

# clean column names (remove "matches_")
df_reg.columns = df_reg.columns.str.replace("^matches_", "", regex=True)


# -------------------------------------------------------------------
# 1. NLTK & evaluate setup
# -------------------------------------------------------------------
print("1. NLTK & evaluate setup")

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

#BERT_MODEL_TYPE = "microsoft/deberta-v3-base"  # smaller, still good
# BERT_MODEL_TYPE = "microsoft/deberta-v3-large"  # medium
BERT_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"  # best, hangs in Ignacio laptop


try:
    from bert_score import score as bert_score
    HAVE_BERTSCORE = True
    print(f"   BERTScore available via bert-score (model_type={BERT_MODEL_TYPE})")
except Exception as e:
    HAVE_BERTSCORE = False
    bert_score = None
    print(f"   WARNING: bert-score not available ({e}). Continuing without it.")

metrics = ["bleu", "rougeL", "meteor", "bertscore"]

# -------------------------------------------------------------------
# 2. Columns to compare between registry and copilot JSONs
# -------------------------------------------------------------------
print("2. Columns to compare between registry and copilot JSONs")
cols_to_compare = [
    'dataset/provenance', 'dataset/splits', 'dataset/redundancy', 'dataset/availability',
    'optimization/algorithm', 'optimization/meta', 'optimization/encoding', 'optimization/parameters',
    'optimization/features', 'optimization/fitting', 'optimization/regularization', 'optimization/config',
    'model/interpretability', 'model/output', 'model/duration', 'model/availability',
    'evaluation/method', 'evaluation/measure', 'evaluation/comparison', 'evaluation/confidence',
    'evaluation/availability'
]

# -------------------------------------------------------------------
# 3. Helper to compute metrics for a single pair of texts
# -------------------------------------------------------------------
print("3. Helper to compute metrics for a single pair of texts")

def clean_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def compute_text_metrics(prediction: str, reference: str, pmcid=None, col=None):
    prediction = (prediction or "").strip()
    reference  = (reference  or "").strip()

    if not prediction or not reference:
        return {"bleu": None, "rougeL": None, "meteor": None, "bertscore": None}

    try:
        bleu_res = bleu_metric.compute(predictions=[prediction], references=[[reference]])
        rouge_res = rouge_metric.compute(predictions=[prediction], references=[reference])
        meteor_res = meteor_metric.compute(predictions=[prediction], references=[reference])
    except Exception as e:
        print(f"[{pmcid}] metric failure col='{col}': {e}")
        return {"bleu": None, "rougeL": None, "meteor": None, "bertscore": None}

    out = {
        "bleu": bleu_res.get("bleu"),
        "rougeL": rouge_res.get("rougeL"),
        "meteor": meteor_res.get("meteor"),
        "bertscore": None
    }

    # from bert-score package:
    try:
        out["bertscore"] = bertscore(prediction, reference)
    except Exception as e:
        print(f"[{pmcid}] BERTScore failed col='{col}': {e}")
        out["bertscore"] = None

    return out




def bertscore(prediction: str, reference: str):
    prediction = (prediction or "").strip()
    reference  = (reference  or "").strip()

    if not prediction or not reference:
        return None

    P, R, F1 = bert_score(
        [prediction],
        [reference],
        model_type=BERT_MODEL_TYPE,
        lang="en",
        rescale_with_baseline=False,  # keep False for stability
        verbose=False
    )

    return float(F1[0].item())


# -------------------------------------------------------------------
# 4. Loop over df_reg rows, load corresponding copilot JSON by PMCID,
#    compare the columns and compute metrics
# -------------------------------------------------------------------
print("4. Loop over df_reg rows, load corresponding copilot JSON by PMCID, compare the columns and compute metrics")

results = []

# Folder to store per-PMCID metric JSONs
registry_copilot_dir = "registry-copilot-longbert"
os.makedirs(registry_copilot_dir, exist_ok=True)

# Filter registry rows that have valid PMCIDs
valid_rows = [
    (idx, row)
    for idx, row in df_reg.iterrows()
    if isinstance(row.get("mapped_pmcid"), str)
    and row.get("mapped_pmcid", "").startswith("PMC")
]

total = len(valid_rows)
print(f"Total registry entries with valid PMCIDs: {total}")

# Loop with progress
for i, (idx, row) in enumerate(valid_rows, start=1):
    pmcid = row.get("mapped_pmcid")
    print(f"\n[{i}/{total}] Processing {pmcid}...")

    copilot_path = os.path.join("copilot-results", f"{pmcid}.json")
    if not os.path.exists(copilot_path):
        print(f"   → Copilot file NOT found for {pmcid}, skipping.")
        continue

    # Load copilot JSON
    try:
        with open(copilot_path, "r", encoding="utf-8") as f:
            copilot = json.load(f)
    except Exception as e:
        print(f"   → ERROR loading {pmcid}: {e}")
        continue

    # Ensure DataFrame
    if isinstance(copilot, dict):
        df_cp = pd.DataFrame([copilot])
    else:
        df_cp = pd.DataFrame(copilot)
        if len(df_cp) > 1:
            df_cp = df_cp.iloc[[0]]

    #df_cp.columns = df_cp.columns.str.replace("^matches_", "", regex=True)

    row_metrics = {
        "pmcid": pmcid,
        "registry_index": idx,
        "publication_pmid": row.get("publication_pmid"),
        "publication_title": row.get("publication_title")
    }

    # Compare columns
    for col in cols_to_compare:
        registry_missing = (
            (col not in df_reg.columns)
            or pd.isna(row.get(col))
            or str(row.get(col)).strip() == ""
        )
        copilot_missing = (
            (col not in df_cp.columns)
            or pd.isna(df_cp.iloc[0].get(col))
            or str(df_cp.iloc[0].get(col)).strip() == ""
        )

        if registry_missing or copilot_missing:
            row_metrics[f"{col}__bleu"] = float("nan")
            row_metrics[f"{col}__rougeL"] = float("nan")
            row_metrics[f"{col}__meteor"] = float("nan")
            if HAVE_BERTSCORE:
                row_metrics[f"{col}__bertscore"] = float("nan")
            continue

        ref_text = row.get(col)
        pred_text = df_cp.iloc[0].get(col)

        metrics = compute_text_metrics(pred_text, ref_text, pmcid=pmcid, col=col)


        row_metrics[f"{col}__bleu"] = metrics["bleu"]
        row_metrics[f"{col}__rougeL"] = metrics["rougeL"]
        row_metrics[f"{col}__meteor"] = metrics["meteor"]
        row_metrics[f"{col}__bertscore"] = metrics["bertscore"]


    results.append(row_metrics)

    #save each entry in JSON
    out_json_path = os.path.join(registry_copilot_dir, f"{pmcid}.json")
    with open(out_json_path, "w", encoding="utf-8") as jf:
        json.dump(row_metrics, jf, indent=2, ensure_ascii=False)

    print(f"   → Metrics saved to {out_json_path}")    

    #print overall progress every 10 items
    if i % 10 == 0:
        print(f"   → Progress: {i}/{total} ({i/total:.1%}) complete")

print("\nFinished processing all PMCs.")

# -------------------------------------------------------------------
# 4b. Extra pass: copilot JSONs that don't have mapped_pmcid in registry
#      but can be matched via publication_pmid
# -------------------------------------------------------------------
print("\n4b. Extra pass: processing copilot JSONs without mapped_pmcid in registry")

copilot_dir = "copilot-results"
if not os.path.isdir(copilot_dir):
    print(f"  WARNING: copilot directory '{copilot_dir}' not found.")
else:
    # All PMCIDs that have a JSON file
    copilot_pmcids = {
        fname[:-5]  # strip ".json"
        for fname in os.listdir(copilot_dir)
        if fname.endswith(".json") and fname.startswith("PMC")
    }

    # PMCIDs already handled via mapped_pmcid
    registry_pmcids = {
        str(p)
        for p in df_reg.get("mapped_pmcid", [])
        if isinstance(p, str) and p.startswith("PMC")
    }

    # PMCIDs that exist as JSON but have NO mapped_pmcid in the registry
    orphan_pmcids = sorted(copilot_pmcids - registry_pmcids)
    print(f"   Found {len(orphan_pmcids)} copilot PMCIDs without mapped_pmcid in registry.")

    # For faster lookup by publication_pmid
    if "publication_pmid" in df_reg.columns:
        reg_by_pmid = df_reg.set_index("publication_pmid")
    else:
        reg_by_pmid = None
        print("   WARNING: 'publication_pmid' not in registry; cannot match orphans by PMID.")

    for j, pmcid in enumerate(orphan_pmcids, start=1):
        print(f"\n   [orphan {j}/{len(orphan_pmcids)}] Processing {pmcid}...")

        copilot_path = os.path.join(copilot_dir, f"{pmcid}.json")
        try:
            with open(copilot_path, "r", encoding="utf-8") as f:
                copilot = json.load(f)
        except Exception as e:
            print(f"      → ERROR loading {pmcid}: {e}")
            continue

        # Turn into DataFrame (1 row)
        if isinstance(copilot, dict):
            df_cp = pd.DataFrame([copilot])
        else:
            df_cp = pd.DataFrame(copilot)
            if len(df_cp) > 1:
                df_cp = df_cp.iloc[[0]]

        # Try to get publication_pmid from copilot
        if "publication_pmid" not in df_cp.columns or reg_by_pmid is None:
            print(f"      → No 'publication_pmid' in copilot or registry; cannot match. Skipping.")
            continue

        pmid_cp = df_cp.iloc[0]["publication_pmid"]
        if pd.isna(pmid_cp) or str(pmid_cp).strip() == "":
            print(f"      → Empty publication_pmid in copilot; skipping.")
            continue

        pmid_cp_str = str(pmid_cp).strip()
        if pmid_cp_str not in reg_by_pmid.index:
            print(f"      → publication_pmid={pmid_cp_str} not found in registry; skipping.")
            continue

        # Take the first matching registry row for this PMID
        row = reg_by_pmid.loc[pmid_cp_str]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        idx = int(row.name) if isinstance(row.name, (int, float)) else -1

        print(f"      → Matched orphan {pmcid} to registry PMID {pmid_cp_str} (index {idx})")

        row_metrics = {
            "pmcid": pmcid,
            "registry_index": idx,
            "publication_pmid": row.get("publication_pmid"),
            "publication_title": row.get("publication_title"),
        }

        # Compare each requested column
        for col in cols_to_compare:

            ref_text = str(row.get(col)).strip()
            pred_text = str(df_cp.iloc[0].get(col)).strip()

            m = compute_text_metrics(prediction=pred_text, reference=ref_text)

            row_metrics[f"{col}__bleu"] = m["bleu"]
            row_metrics[f"{col}__rougeL"] = m["rougeL"]
            row_metrics[f"{col}__meteor"] = m["meteor"]
            row_metrics[f"{col}__bertscore"] = m["bertscore"]

        # Add to global results list
        results.append(row_metrics)

        # Save per-PMCID JSON for orphan
        out_json_path = os.path.join(registry_copilot_dir, f"{pmcid}.json")
        with open(out_json_path, "w", encoding="utf-8") as jf:
            json.dump(row_metrics, jf, indent=2, ensure_ascii=False)

        print(f"      → Orphan metrics saved to {out_json_path}")


# -------------------------------------------------------------------
# 5. Convert metrics to DataFrame and inspect/save
# -------------------------------------------------------------------
print("5. Convert metrics to DataFrame and inspect/save")
metrics_df = pd.DataFrame(results)

#print(metrics_df.head())

# Optional: save to CSV
out_metrics_path = "results/copilot_vs_registry_text_metrics2_long_bert.csv"
os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
metrics_df.to_csv(out_metrics_path, index=False)
print(f"\nMetrics saved to: {out_metrics_path}")

# -------------------------------------------------------------------
# 6.  metrics stats
# -------------------------------------------------------------------
print("6.  metrics stats")
summary_rows = []

for col in cols_to_compare:
    for metric in metrics:
        col_name = f"{col}__{metric}"
        if col_name not in metrics_df.columns:
            continue

        series = metrics_df[col_name].dropna()

        if series.empty:
            continue

        summary_rows.append({
            "column": col,
            "metric": metric,
            "count": series.count(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "25%": series.quantile(0.25),
            "50%": series.median(),
            "75%": series.quantile(0.75),
            "max": series.max(),
        })

stats_df = pd.DataFrame(summary_rows)
stats_df = stats_df.sort_values(["column", "metric"]).reset_index(drop=True)

out_stats_path = "results/metrics_stats2_long_bert.csv"
os.makedirs(os.path.dirname(out_stats_path), exist_ok=True)
stats_df.to_csv(out_stats_path, index=False)

print(stats_df)

# -------------------------------------------------------------------
# 7. Diagnostic report: registry vs copilot vs metrics
# -------------------------------------------------------------------
print("\n7. Diagnostic report: registry vs copilot vs metrics")

# --- Collect PMCIDs from registry ---
if "mapped_pmcid" not in df_reg.columns:
    print("  WARNING: 'mapped_pmcid' column not found in registry dataframe.")
    registry_pmcids = set()
else:
    registry_pmcids = set(
        str(pmcid)
        for pmcid in df_reg["mapped_pmcid"]
        if isinstance(pmcid, str) and pmcid.startswith("PMC")
    )

# --- Collect PMCIDs from copilot-results filenames ---
copilot_dir = "copilot-results"
if not os.path.isdir(copilot_dir):
    print(f"  WARNING: copilot directory '{copilot_dir}' not found.")
    copilot_pmcids = set()
else:
    copilot_pmcids = set(
        fname[:-5]  # strip ".json"
        for fname in os.listdir(copilot_dir)
        if fname.endswith(".json") and fname.startswith("PMC")
    )

# --- Collect PMCIDs from metrics_df ---
if "pmcid" not in metrics_df.columns:
    print("  WARNING: 'pmcid' column not found in metrics_df.")
    metrics_pmcids = set()
else:
    metrics_pmcids = set(
        str(pmcid)
        for pmcid in metrics_df["pmcid"]
        if isinstance(pmcid, str) and pmcid.startswith("PMC")
    )

# --- Set comparisons ---
registry_only = sorted(registry_pmcids - copilot_pmcids)
copilot_only = sorted(copilot_pmcids - registry_pmcids)
both_registry_and_copilot = registry_pmcids & copilot_pmcids
both_but_no_metrics = sorted(both_registry_and_copilot - metrics_pmcids)

print("\n--- High-level counts ---")
print(f"Total registry rows                 : {len(df_reg)}")
print(f"Registry PMCIDs (mapped_pmcid)      : {len(registry_pmcids)}")
print(f"Copilot JSON PMCIDs (files)         : {len(copilot_pmcids)}")
print(f"PMCIDs with computed metrics        : {len(metrics_pmcids)}")

print("\n--- Mismatch breakdown ---")

print(f"1) In registry BUT missing copilot JSON : {len(registry_only)}")
if registry_only:
    print("   Example(s): " + ", ".join(registry_only[:10]))

print(f"2) In copilot JSON BUT NOT in registry  : {len(copilot_only)}")
if copilot_only:
    print("   Example(s): " + ", ".join(copilot_only[:10]))

print(f"3) In BOTH registry & copilot BUT missing in metrics_df : {len(both_but_no_metrics)}")
if both_but_no_metrics:
    print("   Example(s): " + ", ".join(both_but_no_metrics[:10]))

# Optional: also write a detailed text report to a file
report_path = "results/pmcid_diagnostic_report.txt"
with open(report_path, "w", encoding="utf-8") as rep:
    rep.write("Diagnostic report: registry vs copilot \n\n")
    rep.write(f"Total registry rows                 : {len(df_reg)}\n")
    rep.write(f"Registry PMCIDs (mapped_pmcid)      : {len(registry_pmcids)}\n")
    rep.write(f"Copilot JSON PMCIDs (files)         : {len(copilot_pmcids)}\n")
    rep.write(f"PMCIDs with computed metrics        : {len(metrics_pmcids)}\n\n")

    rep.write("1) PMCIDs in registry BUT missing copilot JSON:\n")
    rep.write(f"   Count: {len(registry_only)}\n")
    if registry_only:
        rep.write("   List:\n")
        for pmc in registry_only:
            rep.write(f"     {pmc}\n")
    rep.write("\n")

    rep.write("2) PMCIDs in copilot JSON BUT NOT in registry:\n")
    rep.write(f"   Count: {len(copilot_only)}\n")
    if copilot_only:
        rep.write("   List:\n")
        for pmc in copilot_only:
            rep.write(f"     {pmc}\n")
    rep.write("\n")

    rep.write("3) PMCIDs in BOTH registry & copilot BUT missing in metrics_df:\n")
    rep.write(f"   Count: {len(both_but_no_metrics)}\n")
    if both_but_no_metrics:
        rep.write("   List:\n")
        for pmc in both_but_no_metrics:
            rep.write(f"     {pmc}\n")

print(f"\nDetailed diagnostic report written to: {report_path}")
