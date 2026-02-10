import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "evaluation_results.tsv")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "statistical_analysis_report.txt")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Statistical_Plots")

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        sys.exit(1)
    
    df = pd.read_csv(DATA_FILE, sep='\t')
    return df

def perform_analysis(df):
    # Mapping
    # A = Human, B = Copilot
    # Score: Copilot Better (+1), Tie (0), Human Better (-1)
    
    rank_map = {
        'A_Better': -1,
        'B_Better': 1,
        'Tie_High': 0,
        'Tie_Low': 0
    }
    
    # Filter out rows with no rank
    df = df.dropna(subset=['Rank'])
    df = df[df['Rank'].isin(rank_map.keys())]
    
    # Filter out publication fields as they are not relevant for this analysis
    df = df[~df['Field'].str.startswith('publication/')]

    df['Score'] = df['Rank'].map(rank_map)
    
    results = {}
    
    # --- Global Analysis ---
    scores = df['Score']
    n = len(scores)
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Counts
    counts = df['Rank'].value_counts()
    n_copilot = counts.get('B_Better', 0)
    n_human = counts.get('A_Better', 0)
    n_tie_high = counts.get('Tie_High', 0)
    n_tie_low = counts.get('Tie_Low', 0)
    n_ties = n_tie_high + n_tie_low
    
    # 1. One-sample t-test (Ho: mean = 0)
    t_stat, p_ttest = stats.ttest_1samp(scores, 0)
    
    # 2. Wilcoxon Signed-Rank Test (Ho: median = 0, symmetric distribution)
    # Note: Wilcoxon handles zeros by excluding them usually, or splitting. 
    # scipy's wilcoxon does zero-method='wilcox' (excludes) or 'zsplit' or 'pratt'.
    # For "A vs B", we usually exclude ties for the sign test/signed-rank if looking at preference.
    # However, "Tie" is a valid result here representing equality. 
    # If we strictly test if B is *Better* than A, we compare B_Better vs A_Better counts (Binomial/Sign test).
    
    # Binomial Test (Sign Test) - ignoring Ties
    # Ho: p(Copilot Better) = p(Human Better) = 0.5 (among non-ties)
    n_decisive = n_copilot + n_human
    if n_decisive > 0:
        p_binom = stats.binomtest(n_copilot, n_decisive, p=0.5).pvalue
    else:
        p_binom = 1.0

    # Wilcoxon on full scores (treating 0 as 0) isn't standard because it ranks absolute differences. 
    # abs(0) is 0. 
    # Standard practice for "Is B better than A" with ties allowed:
    # Use Sign Test (Binomial) on the differences.
    
    results['global'] = {
        'n': n,
        'mean': mean_score,
        'counts': {
            'Copilot Better': n_copilot,
            'Human Better': n_human,
            'Tie': n_ties,
            'Tie_High': n_tie_high,
            'Tie_Low': n_tie_low
        },
        'ttest': {'t': t_stat, 'p': p_ttest},
        'binom': {'p': p_binom}
    }
    
    # --- Field Level Analysis ---
    field_stats = []
    for field, group in df.groupby('Field'):
        f_scores = group['Score']
        f_mean = f_scores.mean()
        f_n = len(f_scores)
        
        # T-test per field (if variance > 0)
        if f_scores.std() > 0:
            _, f_p_ttest = stats.ttest_1samp(f_scores, 0)
        else:
            f_p_ttest = 1.0 if f_mean == 0 else 0.0 # If all 1s, significantly diff from 0? technically yes.
            
        field_stats.append({
            'Field': field,
            'Mean_Score': f_mean,
            'N': f_n,
            'P_Value': f_p_ttest,
            'Copilot_Wins': len(f_scores[f_scores == 1]),
            'Human_Wins': len(f_scores[f_scores == -1]),
            'Ties': len(f_scores[f_scores == 0])
        })
    
    results['fields'] = pd.DataFrame(field_stats).sort_values('Mean_Score', ascending=False)
    
    return results, df

def generate_report(results, output_path):
    g = results['global']
    fields_df = results['fields']
    
    lines = []
    lines.append("Statistical Analysis Report: DOME Copilot vs Human Evaluation")
    lines.append("============================================================")
    lines.append("")
    lines.append("## 1. Methodology")
    lines.append("### Data Source")
    lines.append("The analysis is based on the manual evaluation of 30 randomly selected papers.")
    lines.append("The results are loaded from `evaluation_results.tsv`.")
    lines.append("Note: `publication/*` fields (title, year, etc.) were EXCLUDED from this analysis as they are effectively ground-truth extraction tasks where variance is minimal and not indicative of intelligent assistance capability.")
    lines.append("")
    lines.append("### Variable Encoding")
    lines.append("- **Human Better (A_Better)**: -1")
    lines.append("- **Tie (Tie_High/Low)**: 0. Both High Quality (correct extraction) and Low Quality (both hallucinated/wrong) are treated as neutral outcomes (0) for the comparative statistics, as neither the Human nor the Copilot is superior to the other in these cases. However, their distinction is preserved in the detailed counts.")
    lines.append("- **Copilot Better (B_Better)**: +1")
    lines.append("")
    lines.append("### Statistical Tests Performed")
    lines.append("1. **Global One-Sample T-Test**: Tests if the mean preference score across all valid observations is significantly different from 0 (where 0 represents a Tie).")
    lines.append("2. **Binomial Sign Test**: Tests if the proportion of Copilot Wins is significantly different from Human Wins (ignoring ties). Ho: p(Win) = 0.5.")
    lines.append("3. **Per-Field One-Sample T-Test**: Applied to each field individually to determine if the mean score for that specific field is significantly different from 0.")
    lines.append("")
    
    lines.append("## 2. Global Overview")
    lines.append(f"Total Valid Observations (after filtering): {g['n']}")
    lines.append(f"Copilot Better (Wins): {g['counts']['Copilot Better']} ({g['counts']['Copilot Better']/g['n']:.1%})")
    lines.append(f"Human Better (Wins):   {g['counts']['Human Better']} ({g['counts']['Human Better']/g['n']:.1%})")
    lines.append(f"Ties:                  {g['counts']['Tie']} ({g['counts']['Tie']/g['n']:.1%})")
    lines.append(f"    - High Quality Tie (Both Good): {g['counts']['Tie_High']} ({g['counts']['Tie_High']/g['n']:.1%})")
    lines.append(f"    - Low Quality Tie (Both Poor):  {g['counts']['Tie_Low']} ({g['counts']['Tie_Low']/g['n']:.1%})")
    lines.append("")
    lines.append(f"Mean Preference Score (-1 Human, 0 Tie, +1 Copilot): {g['mean']:.3f}")
    lines.append("")
    
    lines.append("## 3. Global Statistical Significance")
    
    lines.append("### A. One-Sample T-Test")
    lines.append("Method: Two-sided test against pop. mean = 0")
    lines.append(f"T-statistic: {g['ttest']['t']:.3f}")
    lines.append(f"P-value: {g['ttest']['p']:.2e}")
    if g['ttest']['p'] < 0.05:
        verdict = "Significant difference found."
        direction = "favoring Copilot" if g['mean'] > 0 else "favoring Human"
    else:
        verdict = "No significant difference found."
        direction = ""
    lines.append(f"Result: {verdict} {direction}")
    lines.append("")
    
    lines.append("### B. Binomial Sign Test")
    lines.append("Method: Exact test, H0: p=0.5 (excluding ties)")
    lines.append(f"P-value: {g['binom']['p']:.2e}")
    if g['binom']['p'] < 0.05:
         lines.append("Result: Reject H0. The win rate is not 50/50.")
    else:
         lines.append("Result: Fail to reject H0. Win rates are comparable.")
    lines.append("\n")
    
    lines.append("## 4. Per-Field Analysis")
    lines.append("Ordering: Descending by Mean Score (Copilot Advantage).\n")
    lines.append("P-Val Column: Result of one-sample t-test for that specific field vs 0 (Tie).\n")
    
    # Format the dataframe for text output
    header = f"{'Field':<35} | {'Mean':<6} | {'Copilot':<8} | {'Human':<6} | {'Tie':<6} | {'P-Val':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for _, row in fields_df.iterrows():
        sig = "*" if row['P_Value'] < 0.05 else ""
        lines.append(f"{row['Field']:<35} | {row['Mean_Score']:>6.3f} | {row['Copilot_Wins']:>8} | {row['Human_Wins']:>6} | {row['Ties']:>6} | {row['P_Value']:>8.1e} {sig}")
        
    lines.append("\n(*) indicates p < 0.05: The preference for one system in this field is statistically significant.")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\nReport generated at: {output_path}")
    print("\nSummary:")
    print("\n".join(lines[:20])) # Print head to console

def create_plots(df, results):
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Global Distribution Plot
    plt.figure(figsize=(10, 6))
    rank_order = ['A_Better', 'Tie_High', 'Tie_Low', 'B_Better']
    # Map for prettier labels
    pretty_labels = {'A_Better': 'Human Better', 'Tie_High': 'Tie (High)', 'Tie_Low': 'Tie (Low)', 'B_Better': 'Copilot Better'}
    
    ax = sns.countplot(data=df, x='Rank', order=rank_order, palette='viridis')
    ax.set_xticklabels([pretty_labels.get(x.get_text()) for x in ax.get_xticklabels()])
    plt.title('Distribution of Evaluation Ranks (Global)')
    plt.xlabel('Result')
    plt.ylabel('Count')
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "global_rank_distribution.png"))
    plt.close()
    
    # 2. Field Performance (Mean Score)
    plt.figure(figsize=(12, 10))
    fields_df = results['fields'].sort_values('Mean_Score', ascending=True) # Sort for horiz plot
    
    colors = ['red' if x < 0 else 'green' for x in fields_df['Mean_Score']]
    
    plt.barh(fields_df['Field'], fields_df['Mean_Score'], color=colors, alpha=0.7)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Copilot Value Added per Field\n(Positive = Copilot Better, Negative = Human Better)')
    plt.xlabel('Mean Preference Score (-1 to +1)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mean_score_by_field.png"))
    plt.close()

    # 3. Stacked Bar of Proportions by Field
    # Prepare data
    cross = pd.crosstab(df['Field'], df['Rank'])
    
    # Ensure columns exist
    for col in ['A_Better', 'Tie_High', 'Tie_Low', 'B_Better']:
        if col not in cross.columns:
            cross[col] = 0
            
    # Combine Ties
    cross['Tie'] = cross['Tie_High'] + cross['Tie_Low']
    
    # Select and Reorder
    plot_data = cross[['A_Better', 'Tie', 'B_Better']]
    
    # Normalize to 100%
    plot_data_pct = plot_data.div(plot_data.sum(axis=1), axis=0) * 100
    
    # Sort by Copilot Score (Using the fields_df order)
    sorted_fields = results['fields'].sort_values('Mean_Score', ascending=True)['Field']
    plot_data_pct = plot_data_pct.reindex(sorted_fields)
    
    # Plot
    ax = plot_data_pct.plot(kind='barh', stacked=True, figsize=(14, 12), 
                            color=['#e74c3c', '#95a5a6', '#2ecc71']) # Red, Grey, Green
    
    plt.title('Result Distribution by Field')
    plt.xlabel('Percentage')
    plt.legend(['Human Better', 'Tie', 'Copilot Better'], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "field_distribution_stacked.png"))
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Performing analysis...")
    results, df_processed = perform_analysis(df)
    
    print("Generating report...")
    generate_report(results, OUTPUT_REPORT)
    
    print("Generatng plots...")
    create_plots(df_processed, results)
    
    print("Done.")
