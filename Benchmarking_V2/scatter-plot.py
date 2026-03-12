import pandas as pd
import numpy as np
import os

pmcids_to_check = [
    "PMC10716825","PMC10730818","PMC10940896","PMC11258913","PMC11659980",
    "PMC11899596","PMC12366053","PMC2752621","PMC3292016","PMC3967921",
    "PMC4058174","PMC4289375","PMC4315323","PMC4589233","PMC4606520",
    "PMC4894951","PMC5034704","PMC5079830","PMC5550971","PMC5650527",
    "PMC5821114","PMC5910428","PMC6436896","PMC6548586","PMC6679781",
    "PMC6851483","PMC7035778","PMC7212484","PMC7692026","PMC7721480",
    "PMC8230313"
]

v2 = pd.read_csv("results/copilot_vs_registry_text_metrics2_new.csv")
v1 = pd.read_csv("results/copilot_vs_registry_text_metrics2_old.csv")

valid_idx = v2["registry_index"].unique()
v1_filtered = v1[v1["registry_index"].isin(valid_idx)].copy()
v1_filtered = v1_filtered.sort_values("pmcid").reset_index(drop=True)
v2 = v2.sort_values("pmcid").reset_index(drop=True)
len(v1_filtered), len(v2)
(v1_filtered["pmcid"].values == v2["pmcid"].values).all()
v1 = v1_filtered.copy()

# remove common with validation dataset
v1= v1[~v1["pmcid"].isin(pmcids_to_check)]
v2= v2[~v2["pmcid"].isin(pmcids_to_check)]


# indices that exist only in v1
extra_idx = [67, 78, 149, 188, 216]

# remove them
v1 = v1[~v1["registry_index"].isin(extra_idx)].copy()

v2_idx = v2["registry_index"].sort_values()
v2_idx.to_csv("v2_registry_index_list.csv", index=False)
v2_idx.to_csv("v2_registry_index_list.txt", index=False, header=False)

# check lengths
print(len(v1), len(v2))


os.makedirs("figures", exist_ok=True)


import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

metrics = ["bleu", "rougeL", "meteor", "bertscore"]
titles  = ["BLEU", "ROUGE-L", "METEOR", "BERTScore"]

# 1) compute global delta range across ALL metrics
all_deltas = []
for metric in metrics:
    cols = [c for c in v1.columns if c.endswith(f"__{metric}")]
    v1_score = v1[cols].mean(axis=1, skipna=True)
    v2_score = v2[cols].mean(axis=1, skipna=True)
    all_deltas.append((v2_score - v1_score).to_numpy())

all_deltas = np.concatenate(all_deltas)
all_deltas = all_deltas[~np.isnan(all_deltas)]

# symmetric bounds
max_abs = np.max(np.abs(all_deltas))

# center at 0 so white = no change
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

sc = None
for ax, metric, title in zip(axes, metrics, titles):
    cols = [c for c in v1.columns if c.endswith(f"__{metric}")]
    v1_score = v1[cols].mean(axis=1, skipna=True)
    v2_score = v2[cols].mean(axis=1, skipna=True)
    delta = v2_score - v1_score

    sc = ax.scatter(
        v1_score, v2_score,
        c=delta,
        cmap="coolwarm",
        norm=norm,          #shared normalization
        alpha=0.7,
        s=18
    )

    ax.plot([0, 0.6], [0, 0.6], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("V1 mean score")
    ax.set_ylabel("V2 mean score")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)

fig.suptitle("Copilot V1 vs V2 (colored by Δ = V2 − V1)", fontsize=14)

cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.9)
cbar.set_label("Δ score (V2 − V1)")

plt.savefig("figures/copilot_v1_vs_v2_scatter_multipanel_delta_191.png", dpi=600, bbox_inches="tight")
plt.savefig("figures/copilot_v1_vs_v2_scatter_multipanel_delta_191.pdf", bbox_inches="tight")
plt.show()




#stats

rows = []

for m in metrics:

    cols = [c for c in v1.columns if c.endswith(f"__{m}")]

    for col in cols:

        group = col.split("/")[0]   # Dataset / Optimization / Model / Evaluation

        delta = v2[col] - v1[col]

        rows.append({
            "group": group,
            "field": col,
            "metric": m,
            "n": int(delta.notna().sum()),
            "%_improved": (delta > 0).mean()*100,
            "median_delta": delta.median()
        })

df_dome = pd.DataFrame(rows)


summary_dome = (
    df_dome
    .groupby(["group","metric"])
    .agg(
        n_fields=("field","count"),
        pct_improved=("%_improved","mean"),
        median_delta=("median_delta","median")
    )
    .reset_index()
)

print(summary_dome.round(3))
