#!/usr/bin/env python3
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "epmc_publication_data.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "epmc_publication_growth.png")

def main():
    years = []
    ml_counts = []
    ai_counts = []
    combined_counts = []

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            years.append(int(row["Year"]))
            ml_counts.append(int(row["ML_Collected"]))
            ai_counts.append(int(row["AI_Collected"]))
            combined_counts.append(int(row["Combined_Deduplicated"]))

    fig, ax = plt.subplots(figsize=(16, 9))

    # ── Combined line (plotted first → top of legend) ──
    ax.plot(years, combined_counts,
            marker="s", linewidth=2.5, linestyle="--",
            label="Combined (AI ∪ ML) — deduplicated",
            color="black", markersize=7, zorder=3)

    # ── Value labels on combined line ──
    for i, (x, y_val) in enumerate(zip(years, combined_counts)):
        # Adjust size and alignment
        fontsize = 13  # Even larger!
        if x >= 2019:
            if x in [2024, 2025]:
                # 9 o'clock (straight left)
                xytext = (-15, 0)
                ha = "right"
                va = "center"
            elif x in [2021, 2022]:
                # 10:30 (top-left angle)
                xytext = (-10, 10)
                ha = "right"
                va = "bottom"
            elif x == 2023:
                # 11:00 (more towards top, slightly left)
                xytext = (-6, 12)
                ha = "right"
                va = "bottom"
            elif x == 2020:
                # 11:15 (mostly top, slight left)
                xytext = (-2, 12)
                ha = "right"
                va = "bottom"
            elif x == 2019:
                # 11:30 (almost straight top, slightly left)
                xytext = (-8, 14)
                ha = "center"
                va = "bottom"
            else:
                xytext = (-15, 15)
                ha = "right"
                va = "bottom"

            ax.annotate(
                f"{y_val:,}",
                xy=(x, y_val),
                xytext=xytext,
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
                ha=ha, va=va,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="none", alpha=0.9),
                zorder=5
            )
        else:
            is_high = (i % 2 == 0)
            if is_high:
                y_offset = 35
                if x == 2018:
                    y_offset = 40
                ax.annotate(
                    f"{y_val:,}",
                    xy=(x, y_val),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=fontsize,
                    fontweight="bold",
                    ha="center", va="bottom",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=1.2, shrinkA=0, shrinkB=4),
                    zorder=5
                )
            else:
                y_offset = 8  # Just above the dot, no line
                ax.annotate(
                    f"{y_val:,}",
                    xy=(x, y_val),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=fontsize,
                    fontweight="bold",
                    ha="center", va="bottom",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="none", alpha=0.9),
                    zorder=5
                )



    # ── Axes ──
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_xticks(range(min(years), max(years) + 1))
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    ax.set_xlabel("Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Publications", fontsize=14, fontweight="bold")


    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to: {OUTPUT_FILE}")
    plt.close(fig)

if __name__ == "__main__":
    main()
