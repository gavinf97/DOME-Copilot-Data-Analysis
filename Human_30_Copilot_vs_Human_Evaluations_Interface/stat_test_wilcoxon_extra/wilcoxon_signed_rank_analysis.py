#!/usr/bin/env python3
"""
wilcoxon_signed_rank_analysis.py

Wilcoxon Signed-Rank Test analysis of DOME Copilot vs Human evaluation data.

Data source: ../Human_30_Copilot_vs_Human_Evaluations_Interface/evaluation_results.tsv
  - 30 publications x 21 DOME fields = 630 field evaluations
  - PMC5550971 is explicitly excluded (partial entry: only 3/21 DOME fields filled)
  - publication/* fields are excluded (metadata extraction, not intelligent assistance)

Encoding:
  A_Better (Human)  -> -1
  Tie_High / Tie_Low ->  0
  B_Better (Copilot) -> +1

The Wilcoxon Signed-Rank Test is applied:
  - Globally: tests whether the median preference score differs from 0
  - Per-field: tests whether each DOME field's median score differs from 0
  - Per-publication: tests whether each paper's score profile differs from 0

Zero-handling: scipy default ('wilcox' method) — rows where score == 0 are discarded
from the ranking step. This is the classical treatment when 0 represents a genuine tie.
Results are reported alongside sign test (binomial) for comparison.

Outputs saved to this folder:
  wilcoxon_report.txt          - Full text report of all test results
  Plots/01_global_scores.png   - Bar chart: global score distribution
  Plots/02_field_w_stat.png    - Wilcoxon W-statistic per field (sorted)
  Plots/03_field_pvalue.png    - -log10(p-value) per field with significance line
  Plots/04_field_effect.png    - Effect size (r = Z/sqrt(N)) per field
  Plots/05_per_pub_pvalue.png  - Per-publication p-value distribution
  Plots/06_combined_field.png  - Combined panel: mean score, W-stat, effect size
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(SCRIPT_DIR, '..', 'Human_30_Copilot_vs_Human_Evaluations_Interface',
                           'evaluation_results.tsv')
PLOTS_DIR   = os.path.join(SCRIPT_DIR, 'Plots')
REPORT_FILE = os.path.join(SCRIPT_DIR, 'wilcoxon_report.txt')

os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING & FILTERING
# =============================================================================
RANK_MAP = {
    'A_Better': -1,   # Human better
    'Tie_High':  0,
    'Tie_Low':   0,
    'B_Better':  1,   # Copilot better
}

def load_and_filter():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"ERROR: Data file not found at:\n  {os.path.abspath(DATA_FILE)}\n"
                 "Point DATA_FILE to the correct path.")

    df = pd.read_csv(DATA_FILE, sep='\t')
    print(f"Loaded {len(df)} rows from {os.path.abspath(DATA_FILE)}")

    # Keep only rows with a valid rank
    df = df.dropna(subset=['Rank'])
    df = df[df['Rank'].isin(RANK_MAP.keys())]

    # Exclude PMC5550971 — partial entry (only 3 of 21 DOME fields filled)
    df = df[df['PMCID'] != 'PMC5550971']

    # Exclude publication/* metadata fields
    df = df[~df['Field'].str.startswith('publication/')]

    # Integrity check
    n_pubs   = df['PMCID'].nunique()
    n_fields = df['Field'].nunique()
    n_rows   = len(df)
    assert n_pubs == 30,   f"Expected 30 PMCIDs, got {n_pubs}"
    assert n_fields == 21, f"Expected 21 DOME fields, got {n_fields}"
    assert n_rows == 630,  f"Expected 630 observations (30x21), got {n_rows}"

    df = df.copy()
    df['Score'] = df['Rank'].map(RANK_MAP)

    print(f"After filtering: {n_pubs} publications x {n_fields} fields = {n_rows} observations")
    return df

# =============================================================================
# WILCOXON HELPER
# =============================================================================
def wilcoxon_1samp(scores):
    """
    One-sample Wilcoxon Signed-Rank Test against median = 0.
    Returns (statistic, p_value, n_used, effect_r).
    n_used = observations after removing zeros (classical 'wilcox' zero-method).
    effect_r = r = Z / sqrt(N_used)  (matched-pairs effect size, -1..1)
    """
    arr = np.asarray(scores)
    nonzero = arr[arr != 0]
    n_nonzero = len(nonzero)

    if n_nonzero < 5:
        return np.nan, np.nan, n_nonzero, np.nan

    result = stats.wilcoxon(arr, zero_method='wilcox', alternative='two-sided',
                            method='auto')
    w_stat = result.statistic
    p_val  = result.pvalue

    # Approximate Z from W for effect size
    # E[W] = n*(n+1)/4,  Var[W] = n*(n+1)*(2n+1)/24
    n = n_nonzero
    mean_w = n * (n + 1) / 4
    var_w  = n * (n + 1) * (2 * n + 1) / 24
    z_approx = (w_stat - mean_w) / np.sqrt(var_w)
    effect_r = z_approx / np.sqrt(n)

    return w_stat, p_val, n_nonzero, effect_r

# =============================================================================
# ANALYSIS
# =============================================================================
def run_global(df):
    scores = df['Score'].values
    w, p, n_used, eff_r = wilcoxon_1samp(scores)
    n_total = len(scores)
    n_copilot = (scores ==  1).sum()
    n_human   = (scores == -1).sum()
    n_tie     = (scores ==  0).sum()

    # Binomial sign test for comparison
    n_decisive = n_copilot + n_human
    p_binom = stats.binomtest(int(n_copilot), int(n_decisive), p=0.5).pvalue if n_decisive else 1.0

    return {
        'n_total': n_total, 'n_nonzero': n_used,
        'n_copilot': n_copilot, 'n_human': n_human, 'n_tie': n_tie,
        'mean': scores.mean(), 'median': np.median(scores),
        'W': w, 'p_wilcoxon': p, 'effect_r': eff_r,
        'p_binom': p_binom,
    }

def run_per_field(df):
    rows = []
    for field, grp in df.groupby('Field'):
        scores = grp['Score'].values
        w, p, n_used, eff_r = wilcoxon_1samp(scores)
        n_c = (scores ==  1).sum()
        n_h = (scores == -1).sum()
        n_t = (scores ==  0).sum()
        rows.append({
            'Field': field,
            'N': len(scores), 'N_nonzero': n_used,
            'Mean_Score': scores.mean(),
            'Median_Score': np.median(scores),
            'N_Copilot': n_c, 'N_Human': n_h, 'N_Tie': n_t,
            'W_stat': w, 'P_Value': p, 'Effect_r': eff_r,
        })
    field_df = pd.DataFrame(rows).sort_values('Mean_Score', ascending=False)
    # BH FDR correction
    from statsmodels.stats.multitest import multipletests
    valid     = field_df['P_Value'].notna()
    pvals_arr = field_df.loc[valid, 'P_Value'].values
    _, p_adj, _, _ = multipletests(pvals_arr, method='fdr_bh')
    field_df.loc[valid, 'P_Adj_BH'] = p_adj
    return field_df

def run_per_publication(df):
    rows = []
    for pmcid, grp in df.groupby('PMCID'):
        scores = grp['Score'].values
        w, p, n_used, eff_r = wilcoxon_1samp(scores)
        rows.append({
            'PMCID': pmcid,
            'N': len(scores), 'N_nonzero': n_used,
            'Mean_Score': scores.mean(),
            'W_stat': w, 'P_Value': p, 'Effect_r': eff_r,
        })
    return pd.DataFrame(rows).sort_values('Mean_Score', ascending=False)

# =============================================================================
# PLOTS
# =============================================================================
sns.set_theme(style='whitegrid')
ALPHA = 0.05

def plot_global_scores(df):
    rank_map_labels = {
        'A_Better': 'Human Better',
        'Tie_High':  'Tie (High Quality)',
        'Tie_Low':   'Tie (Low Quality)',
        'B_Better':  'Copilot Better',
    }
    df2 = df.copy()
    df2['Result'] = df2['Rank'].map(rank_map_labels)

    order = ['Copilot Better', 'Human Better', 'Tie (High Quality)', 'Tie (Low Quality)']
    palette = {
        'Copilot Better':    '#ff7f0e',
        'Human Better':      '#1f77b4',
        'Tie (High Quality)':'#2ca02c',
        'Tie (Low Quality)': '#d62728',
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df2, x='Result', order=order,
                  hue='Result', palette=palette, legend=False, ax=ax)
    ax.set_title('Global Evaluation Result Distribution\n(30 publications × 21 DOME fields = 630 observations)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Result', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    for container in ax.containers:
        ax.bar_label(container, fontsize=11)
    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '01_global_scores.png')
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved: {fp}")

def plot_field_w_stat(field_df):
    sdf = field_df.sort_values('W_stat', ascending=True).dropna(subset=['W_stat'])
    sig = sdf['P_Value'] < ALPHA
    colors = ['#e74c3c' if s else '#95a5a6' for s in sig]

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(sdf['Field'], sdf['W_stat'], color=colors)
    ax.set_title('Wilcoxon W-Statistic per DOME Field\n(red = p < 0.05)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('W Statistic', fontsize=12, fontweight='bold')
    ax.set_ylabel('DOME Field', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '02_field_w_stat.png')
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved: {fp}")

def plot_field_pvalue(field_df):
    sdf = field_df.sort_values('Mean_Score', ascending=True).dropna(subset=['P_Value'])
    log_p  = -np.log10(sdf['P_Value'].clip(lower=1e-15))
    log_pa = -np.log10(sdf['P_Adj_BH'].clip(lower=1e-15))
    threshold = -np.log10(ALPHA)

    x = np.arange(len(sdf))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 9))
    bars1 = ax.barh(x - width/2, log_p,  width, label='Raw p-value',  color='#2e86c1', alpha=0.85)
    bars2 = ax.barh(x + width/2, log_pa, width, label='BH-adjusted p', color='#e67e22', alpha=0.85)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.2,
               label=f'p = {ALPHA} threshold')
    ax.set_yticks(x)
    ax.set_yticklabels(sdf['Field'])
    ax.set_title('Wilcoxon Signed-Rank Test: Significance per DOME Field\n(−log₁₀ p-value; higher = more significant)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('−log₁₀(p-value)', fontsize=12, fontweight='bold')
    ax.set_ylabel('DOME Field', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '03_field_pvalue.png')
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved: {fp}")

def plot_field_effect(field_df):
    sdf = field_df.sort_values('Effect_r', ascending=True).dropna(subset=['Effect_r'])
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in sdf['Effect_r']]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(sdf['Field'], sdf['Effect_r'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.9)
    ax.axvline( 0.3, color='grey', linewidth=0.7, linestyle=':', label='|r|=0.3 (medium)')
    ax.axvline(-0.3, color='grey', linewidth=0.7, linestyle=':')
    ax.axvline( 0.5, color='grey', linewidth=0.7, linestyle='--', label='|r|=0.5 (large)')
    ax.axvline(-0.5, color='grey', linewidth=0.7, linestyle='--')
    ax.set_title('Effect Size (r = Z/√N) per DOME Field\n(positive = Copilot advantage)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Effect Size r', fontsize=12, fontweight='bold')
    ax.set_ylabel('DOME Field', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '04_field_effect.png')
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved: {fp}")

def plot_per_pub_pvalue(pub_df):
    pvals = pub_df['P_Value'].dropna().values
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Histogram of p-values
    axes[0].hist(pvals, bins=15, color='#2e86c1', edgecolor='white', alpha=0.85)
    axes[0].axvline(ALPHA, color='red', linestyle='--', label=f'p={ALPHA}')
    axes[0].set_title('Distribution of Per-Publication\nWilcoxon p-values', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('p-value', fontsize=11)
    axes[0].set_ylabel('Count of Publications', fontsize=11)
    axes[0].legend()

    # Bar chart sorted by mean score
    sdf = pub_df.sort_values('Mean_Score', ascending=True)
    sig = sdf['P_Value'] < ALPHA
    colors = ['#e74c3c' if s else '#95a5a6' for s in sig]
    axes[1].barh(range(len(sdf)), sdf['Mean_Score'], color=colors)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_yticks(range(len(sdf)))
    axes[1].set_yticklabels(sdf['PMCID'], fontsize=7)
    axes[1].set_title('Mean Score per Publication\n(red = p < 0.05 Wilcoxon)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mean Score (−1 Human → +1 Copilot)', fontsize=11)

    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '05_per_pub_pvalue.png')
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved: {fp}")

def plot_combined_field(field_df):
    sdf = field_df.sort_values('Mean_Score', ascending=True).dropna(subset=['P_Value'])
    fields = sdf['Field'].values

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(1, 3, wspace=0.45)

    # Panel A: Mean Score
    ax1 = fig.add_subplot(gs[0])
    colors_a = ['#2ca02c' if v > 0 else '#d62728' for v in sdf['Mean_Score']]
    ax1.barh(fields, sdf['Mean_Score'], color=colors_a, alpha=0.85)
    ax1.axvline(0, color='black', linewidth=0.9)
    ax1.set_xlabel('Mean Score', fontsize=10, fontweight='bold')
    ax1.set_title('A  Mean Score\n(+1 Copilot, −1 Human)', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=8)

    # Panel B: -log10(p)
    ax2 = fig.add_subplot(gs[1])
    log_p  = -np.log10(sdf['P_Value'].clip(lower=1e-15))
    log_pa = -np.log10(sdf['P_Adj_BH'].clip(lower=1e-15))
    threshold = -np.log10(ALPHA)
    x = np.arange(len(sdf))
    width = 0.38
    ax2.barh(x - width/2, log_p,  width, color='#2e86c1', alpha=0.85, label='Raw')
    ax2.barh(x + width/2, log_pa, width, color='#e67e22', alpha=0.85, label='BH-adj')
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=1.0)
    ax2.set_yticks(x)
    ax2.set_yticklabels(fields, fontsize=8)
    ax2.set_xlabel('−log₁₀(p-value)', fontsize=10, fontweight='bold')
    ax2.set_title('B  Wilcoxon Significance\n(dashed = p=0.05)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)

    # Panel C: Effect size r
    ax3 = fig.add_subplot(gs[2])
    colors_c = ['#2ca02c' if v > 0 else '#d62728' for v in sdf['Effect_r'].fillna(0)]
    ax3.barh(fields, sdf['Effect_r'].fillna(0), color=colors_c, alpha=0.85)
    ax3.axvline(0,    color='black', linewidth=0.9)
    ax3.axvline( 0.5, color='grey',  linewidth=0.7, linestyle='--', label='large (|r|=0.5)')
    ax3.axvline(-0.5, color='grey',  linewidth=0.7, linestyle='--')
    ax3.set_xlabel('Effect Size r', fontsize=10, fontweight='bold')
    ax3.set_title('C  Effect Size r = Z/√N\n(positive = Copilot advantage)', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.tick_params(axis='y', labelsize=8)

    fig.suptitle('Wilcoxon Signed-Rank Test: DOME Field Analysis\n'
                 '(30 publications × 21 DOME fields = 630 observations)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, '06_combined_field.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fp}")

# =============================================================================
# REPORT
# =============================================================================
def write_report(global_res, field_df, pub_df):
    g = global_res
    lines = []
    lines.append("Wilcoxon Signed-Rank Test Report: DOME Copilot vs Human Evaluation")
    lines.append("=" * 70)
    lines.append("")
    lines.append("## 1. Data & Methodology")
    lines.append(f"Data file : {os.path.abspath(DATA_FILE)}")
    lines.append(f"Total observations: {g['n_total']} (30 publications × 21 DOME fields)")
    lines.append("PMC5550971 excluded: partial entry with only 3/21 DOME fields filled.")
    lines.append("publication/* fields excluded: metadata extraction fields (title, authors, etc.)")
    lines.append("")
    lines.append("### Encoding")
    lines.append("  A_Better (Human Better)  →  -1")
    lines.append("  Tie_High / Tie_Low       →   0")
    lines.append("  B_Better (Copilot Better)→  +1")
    lines.append("")
    lines.append("### Test: One-Sample Wilcoxon Signed-Rank Test")
    lines.append("  H₀: median preference score = 0 (no systematic preference)")
    lines.append("  H₁: median ≠ 0 (two-sided)")
    lines.append("  Zero handling: 'wilcox' method — scores of 0 are discarded from ranking")
    lines.append("                 (classical treatment for genuine ties)")
    lines.append("  Effect size: r = Z / √N_nonzero  (ranges −1 to +1)")
    lines.append("  Field-level FDR correction: Benjamini-Hochberg (BH)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("## 2. Global Wilcoxon Test")
    lines.append(f"  Total observations          : {g['n_total']}")
    lines.append(f"  Non-zero observations (used): {g['n_nonzero']}")
    lines.append(f"  Zeros (ties, excluded)       : {g['n_tie']}")
    lines.append(f"  Copilot Better (score = +1) : {g['n_copilot']} ({g['n_copilot']/g['n_total']:.1%})")
    lines.append(f"  Human Better   (score = -1) : {g['n_human']}  ({g['n_human']/g['n_total']:.1%})")
    lines.append(f"  Mean score                  : {g['mean']:.4f}")
    lines.append(f"  Median score                : {g['median']:.4f}")
    lines.append("")
    lines.append(f"  Wilcoxon W statistic        : {g['W']:.1f}")
    lines.append(f"  p-value (Wilcoxon)          : {g['p_wilcoxon']:.4e}")
    lines.append(f"  Effect size r               : {g['effect_r']:.4f}")
    sig = g['p_wilcoxon'] < ALPHA if not np.isnan(g['p_wilcoxon']) else False
    if sig:
        direction = "Copilot" if g['mean'] > 0 else "Human"
        lines.append(f"  Result: SIGNIFICANT (p < {ALPHA}) — systematic preference favouring {direction}")
    else:
        lines.append(f"  Result: NOT SIGNIFICANT (p ≥ {ALPHA})")
    lines.append("")
    lines.append(f"  [Reference] Binomial Sign Test p-value: {g['p_binom']:.4e}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("## 3. Per-Field Wilcoxon Test (sorted by Mean Score descending)")
    lines.append("   BH = Benjamini-Hochberg FDR-adjusted p-value")
    lines.append("   * = raw p < 0.05;  ** = BH-adjusted p < 0.05")
    lines.append("")
    hdr = f"{'Field':<35} | {'Mean':>6} | {'N':>3} | {'Nz':>3} | {'W':>8} | {'p-raw':>9} | {'p-BH':>9} | {'r':>6} | Sig"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, row in field_df.iterrows():
        p_raw = row['P_Value']
        p_bh  = row.get('P_Adj_BH', np.nan)
        r_val = row['Effect_r']
        sig_raw = '*'  if (not np.isnan(p_raw) and p_raw < ALPHA) else ' '
        sig_bh  = '*'  if (not np.isnan(p_bh)  and p_bh  < ALPHA) else ' '
        sig_str = sig_raw + sig_bh
        p_raw_s = f"{p_raw:.2e}" if not np.isnan(p_raw) else "   n/a  "
        p_bh_s  = f"{p_bh:.2e}"  if not np.isnan(p_bh)  else "   n/a  "
        r_s     = f"{r_val:.3f}" if not np.isnan(r_val)  else "  n/a "
        w_s     = f"{row['W_stat']:.1f}" if not np.isnan(row['W_stat']) else "    n/a "
        lines.append(
            f"{row['Field']:<35} | {row['Mean_Score']:>6.3f} | {int(row['N']):>3} | {int(row['N_nonzero']) if not np.isnan(row['N_nonzero']) else '  ?':>3} | "
            f"{w_s:>8} | {p_raw_s:>9} | {p_bh_s:>9} | {r_s:>6} | {sig_str}"
        )
    lines.append("")
    lines.append("  * raw p < 0.05    ** BH-adjusted p < 0.05")
    lines.append("")
    n_sig_raw = field_df['P_Value'].lt(ALPHA).sum()
    n_sig_bh  = field_df['P_Adj_BH'].lt(ALPHA).sum() if 'P_Adj_BH' in field_df else 0
    lines.append(f"  Significant fields (raw p < 0.05):    {n_sig_raw} / {len(field_df)}")
    lines.append(f"  Significant fields (BH-adj p < 0.05): {n_sig_bh} / {len(field_df)}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("## 4. Per-Publication Wilcoxon Test (sorted by Mean Score descending)")
    lines.append("   Each publication has 21 field-level scores tested against median = 0")
    lines.append("")
    hdr2 = f"{'PMCID':<15} | {'Mean':>6} | {'N':>3} | {'Nz':>3} | {'W':>8} | {'p-raw':>9} | {'r':>6} | Sig"
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for _, row in pub_df.iterrows():
        p_raw = row['P_Value']
        r_val = row['Effect_r']
        sig_s = '*' if (not np.isnan(p_raw) and p_raw < ALPHA) else ' '
        p_s   = f"{p_raw:.2e}" if not np.isnan(p_raw) else "   n/a  "
        r_s   = f"{r_val:.3f}" if not np.isnan(r_val) else "  n/a "
        w_s   = f"{row['W_stat']:.1f}" if not np.isnan(row['W_stat']) else "    n/a "
        lines.append(
            f"{row['PMCID']:<15} | {row['Mean_Score']:>6.3f} | {int(row['N']):>3} | "
            f"{int(row['N_nonzero']) if not np.isnan(row['N_nonzero']) else '  ?':>3} | "
            f"{w_s:>8} | {p_s:>9} | {r_s:>6} | {sig_s}"
        )
    n_sig_pub = pub_df['P_Value'].lt(ALPHA).sum()
    lines.append("")
    lines.append(f"  Significant publications (raw p < 0.05): {n_sig_pub} / {len(pub_df)}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("## 5. Interpretation Notes")
    lines.append("  - The Wilcoxon Signed-Rank Test is non-parametric and does not assume")
    lines.append("    normality. It tests whether the median score distribution is symmetric")
    lines.append("    around zero.")
    lines.append("  - Effect size r ≈ 0.1 small, 0.3 medium, 0.5 large (Cohen 1988).")
    lines.append("  - For field-level tests, BH-adjusted p-values control the false discovery")
    lines.append("    rate at 5% across the 21 simultaneous comparisons.")
    lines.append("  - Per-publication tests use n=21 observations each; low power for")
    lines.append("    individual publications; interpret as exploratory.")

    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nReport saved: {REPORT_FILE}")
    # Print summary to console
    print("\n--- Summary ---")
    print('\n'.join(lines[:55]))

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    df = load_and_filter()

    print("\n--- Running global Wilcoxon test ---")
    global_res = run_global(df)

    print("--- Running per-field Wilcoxon tests (with BH FDR correction) ---")
    field_df = run_per_field(df)

    print("--- Running per-publication Wilcoxon tests ---")
    pub_df = run_per_publication(df)

    print("\n--- Generating plots ---")
    plot_global_scores(df)
    plot_field_w_stat(field_df)
    plot_field_pvalue(field_df)
    plot_field_effect(field_df)
    plot_per_pub_pvalue(pub_df)
    plot_combined_field(field_df)

    write_report(global_res, field_df, pub_df)

    print("\nDone. All outputs in:", SCRIPT_DIR)
