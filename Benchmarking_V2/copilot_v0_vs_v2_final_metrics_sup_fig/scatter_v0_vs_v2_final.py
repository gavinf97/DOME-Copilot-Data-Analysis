"""
Final version of the Copilot V0 vs V2 scatter multipanel delta figure.
- Relabels 'V1' as 'V0' throughout (x-axis labels, title, colorbar)
- Axis titles are bold, slightly larger (13pt), with increased labelpad
  matching the style of the Human evaluation stacked bar figures.
- Does NOT modify or overwrite any files in Benchmarking_V2/v2/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V0_CSV = os.path.join(SCRIPT_DIR, "..", "Benchmarking_V0_Deprecated", "v1",
                      "copilot_vs_registry_text_metrics2.csv")
V2_CSV = os.path.join(SCRIPT_DIR, "..", "Benchmarking_V0_Deprecated", "v2",
                      "copilot_vs_registry_text_metrics2_new.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PMCIDs shared with the validation / human-evaluation dataset (excluded)
# ---------------------------------------------------------------------------
pmcids_to_check = [
    "PMC10716825", "PMC10730818", "PMC10940896", "PMC11258913", "PMC11659980",
    "PMC11899596", "PMC12366053", "PMC2752621",  "PMC3292016",  "PMC3967921",
    "PMC4058174",  "PMC4289375",  "PMC4315323",  "PMC4589233",  "PMC4606520",
    "PMC4894951",  "PMC5034704",  "PMC5079830",  "PMC5550971",  "PMC5650527",
    "PMC5821114",  "PMC5910428",  "PMC6436896",  "PMC6548586",  "PMC6679781",
    "PMC6851483",  "PMC7035778",  "PMC7212484",  "PMC7692026",  "PMC7721480",
    "PMC8230313",
]

# ---------------------------------------------------------------------------
# Load & align data
# ---------------------------------------------------------------------------
v2 = pd.read_csv(V2_CSV)
v0 = pd.read_csv(V0_CSV)

valid_idx = v2["registry_index"].unique()
v0_filtered = v0[v0["registry_index"].isin(valid_idx)].copy()
v0_filtered = v0_filtered.sort_values("pmcid").reset_index(drop=True)
v2 = v2.sort_values("pmcid").reset_index(drop=True)

v0 = v0_filtered.copy()

# Remove PMCIDs shared with the validation dataset
v0 = v0[~v0["pmcid"].isin(pmcids_to_check)]
v2 = v2[~v2["pmcid"].isin(pmcids_to_check)]

# Registry indices that exist only in v0 (no counterpart in v2)
extra_idx = [67, 78, 149, 188, 216]
v0 = v0[~v0["registry_index"].isin(extra_idx)].copy()

print(f"v0 rows: {len(v0)}  |  v2 rows: {len(v2)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
metrics = ["bleu", "rougeL", "meteor", "bertscore"]
titles  = ["BLEU", "ROUGE-L", "METEOR", "BERTScore"]

# Shared delta normalisation across all metrics
all_deltas = []
for metric in metrics:
    cols = [c for c in v0.columns if c.endswith(f"__{metric}")]
    v0_score = v0[cols].mean(axis=1, skipna=True)
    v2_score = v2[cols].mean(axis=1, skipna=True)
    all_deltas.append((v2_score - v0_score).to_numpy())

all_deltas = np.concatenate(all_deltas)
all_deltas = all_deltas[~np.isnan(all_deltas)]
max_abs = np.max(np.abs(all_deltas))
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

# Axis-label styling to match the stacked-bar evaluation figures
AXIS_LABEL_STYLE = dict(fontsize=13, fontweight="bold", labelpad=10)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

sc = None
for ax, metric, title in zip(axes, metrics, titles):
    cols = [c for c in v0.columns if c.endswith(f"__{metric}")]
    v0_score = v0[cols].mean(axis=1, skipna=True)
    v2_score = v2[cols].mean(axis=1, skipna=True)
    delta = v2_score - v0_score

    sc = ax.scatter(
        v0_score, v2_score,
        c=delta,
        cmap="coolwarm",
        norm=norm,
        alpha=0.7,
        s=18,
    )

    ax.plot([0, 0.6], [0, 0.6], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("V0 mean score", **AXIS_LABEL_STYLE)
    ax.set_ylabel("V2 mean score", **AXIS_LABEL_STYLE)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)

fig.suptitle("Copilot V0 vs V2 (colored by \u0394 = V2 \u2212 V0)", fontsize=14)

cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.9)
cbar.set_label("\u0394 score (V2 \u2212 V0)")

out_png = os.path.join(OUT_DIR, "copilot_v0_vs_v2_scatter_multipanel_delta_191_final.png")
out_pdf = os.path.join(OUT_DIR, "copilot_v0_vs_v2_scatter_multipanel_delta_191_final.pdf")
plt.savefig(out_png, dpi=600, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved:\n  {out_png}\n  {out_pdf}")
plt.show()
