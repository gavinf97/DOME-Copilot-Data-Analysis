#!/usr/bin/env python3
"""
generate_evaluation_analysis_plots.py

Converts the analysis logic from 'Analysis_Evaluation_Results.ipynb' into a script.
Generates evaluation plots and saves them to the 'Evaluation_Plots' directory.
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OS_CWD = os.getcwd()

# Define Output Directory for Plots
PLOTS_DIR = os.path.join(SCRIPT_DIR, "Evaluation_Plots")

# Input Files
EVAL_RESULTS_FILE = os.path.join(SCRIPT_DIR, 'evaluation_results.tsv')

# Directory defaults from notebook (adjusted to be robust)
REGISTRY_DIR = os.path.join(SCRIPT_DIR, "../DOME_Registry_JSON_Files")
RAW_REVIEWS_FILE = "dome_review_raw_human_20260128.json"
USERS_FILE = "dome_users_20260130.json"

# Attempt to locate the data dir 
# (Notebook used '../30_human_evaluation', but repo structure suggests it might be different now)
POSSIBLE_DATA_DIRS = [
    os.path.join(SCRIPT_DIR, "../30_human_evaluation"),
    os.path.join(SCRIPT_DIR, "30_Evaluation_Source_JSONs_Human_and_Copilot_Including_PDFs"),
    os.path.join(SCRIPT_DIR, "30_human_evaluation"),
]
DATA_DIR = None
for d in POSSIBLE_DATA_DIRS:
    if os.path.exists(d):
        DATA_DIR = d
        print(f"Found Data Directory: {DATA_DIR}")
        break

if not DATA_DIR:
    print("WARNING: Could not find the '30_human_evaluation' data directory. Diversity analysis will be skipped.")

# ==============================================================================
# SETUP
# ==============================================================================
def setup_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}")
    else:
        print(f"Output directory exists: {PLOTS_DIR}")

    # Set Style
    sns.set_theme(style="whitegrid")

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def load_evaluation_results():
    if not os.path.exists(EVAL_RESULTS_FILE):
        print(f"Error: {EVAL_RESULTS_FILE} not found.")
        return None
    
    df = pd.read_csv(EVAL_RESULTS_FILE, sep='\t')
    print(f"Total evaluations loaded: {len(df)}")
    return df

def plot_overall_rank_distribution(df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Rank', order=['A_Better', 'B_Better', 'Tie_High', 'Tie_Low'], palette='viridis')
    plt.title('Overall Evaluation Results: Human (A) vs Copilot (B)')
    plt.ylabel('Count')

    # Add labels
    for container in ax.containers:
        ax.bar_label(container)

    filepath = os.path.join(PLOTS_DIR, '01_Overall_Rank_Distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def categorize_result(rank):
    if rank == 'A_Better': return 'Human Wins'
    if rank == 'B_Better': return 'Copilot Wins'
    return 'Tie'

def plot_win_rate_analysis(df):
    df['Outcome'] = df['Rank'].apply(categorize_result)
    outcome_counts = df['Outcome'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgray', '#66b3ff', '#ff9999'])
    plt.title('Win Rate Analysis')
    
    filepath = os.path.join(PLOTS_DIR, '02_Win_Rate_Pie_Chart.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def plot_performance_by_field(df):
    # Filter out publication fields
    df_filtered = df[~df['Field'].str.startswith('publication/')]
    
    plt.figure(figsize=(12, 10))
    sns.histplot(data=df_filtered, y='Field', hue='Rank', multiple='stack', hue_order=['Tie_High', 'Tie_Low', 'A_Better', 'B_Better'], palette='viridis')
    plt.title('Evaluation Results by Field (Excluding Publication Info)')
    plt.xlabel('Count')
    
    filepath = os.path.join(PLOTS_DIR, '03_Performance_By_Field_Stacked.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def plot_copilot_win_rate_per_field(df):
    # Filter out publication fields
    df_filtered = df[~df['Field'].str.startswith('publication/')]

    # Calculate B_Better percentage per field
    field_stats = df_filtered.groupby('Field')['Rank'].value_counts(normalize=True).unstack().fillna(0)
    
    if 'B_Better' in field_stats.columns:
        field_stats['Copilot_Win_Rate'] = field_stats['B_Better'] * 100
    else:
        field_stats['Copilot_Win_Rate'] = 0
        
    field_stats = field_stats.sort_values('Copilot_Win_Rate', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=field_stats['Copilot_Win_Rate'], y=field_stats.index, palette='Blues_r')
    plt.title('Copilot Win Rate by Field (%) (Excluding Publication Info)')
    plt.xlabel('Win Rate (%)')
    
    filepath = os.path.join(PLOTS_DIR, '04_Copilot_Win_Rate_By_Field.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")

def run_diversity_analysis(doi_to_oid, oid_to_user):
    if not DATA_DIR:
        return
        
    print("\n--- Running Diversity Analysis ---")
    
    stats_list = []
    # Search for PMC folders
    pmc_pattern = os.path.join(DATA_DIR, "PMC*")
    pmc_folders = glob.glob(pmc_pattern)

    print(f"Processing {len(pmc_folders)} folders in {DATA_DIR}...")

    for folder in pmc_folders:
        pmcid = os.path.basename(folder)
        human_json_path = os.path.join(folder, f"{pmcid}_human.json")
        main_pdf_path = os.path.join(folder, f"{pmcid}_main.pdf")
        
        # Defaults
        journal = "Unknown"
        curator = "Unknown"
        supp_count = 0
        pdf_size_mb = 0
        
        # Read Metadata from Human JSON
        if os.path.exists(human_json_path):
            try:
                with open(human_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    journal = data.get('publication/journal', 'Unknown')
                    doi = data.get('publication/doi', '').strip()
                    
                    # Resolve Curator
                    if doi in doi_to_oid:
                        oid = doi_to_oid[doi]
                        # curator = oid_to_user.get(oid, "Unknown ID") # Map to name
                        # Just use ID or resolved name if available, but let's stick to Name for readability if possible
                        # The user asked to show "number of publications annotated by the curator"
                        curator = oid_to_user.get(oid, "Assigned Curator") 

            except:
                pass
                
        # Count Supplementaries
        pdfs = glob.glob(os.path.join(folder, "*.pdf"))
        # Exclude main PDF
        supp_pdfs = [p for p in pdfs if os.path.abspath(p) != os.path.abspath(main_pdf_path)]
        supp_count = len(supp_pdfs)
        
        # PDF Size
        if os.path.exists(main_pdf_path):
            pdf_size_mb = os.path.getsize(main_pdf_path) / (1024 * 1024)
            
        stats_list.append({
            'PMCID': pmcid,
            'Journal': journal,
            'Curator': curator,
            'Supplementary_Files_Count': supp_count,
            'Main_PDF_Size_MB': pdf_size_mb
        })

    if not stats_list:
        print("No stats collected. Skipping diversity plots.")
        return

    df_stats = pd.DataFrame(stats_list)

    # Visualization
    sns.set_style("whitegrid")

    # --- Plot 05: Diversity (Journal + Curator) ---
    fig5 = plt.figure(figsize=(16, 12)) 
    gs5 = fig5.add_gridspec(2, 1, height_ratios=[1, 1])

    # Plot 1: Journal Distribution (ALL)
    ax1 = fig5.add_subplot(gs5[0])
    journal_counts = df_stats['Journal'].value_counts()
    sns.barplot(x=journal_counts.values, y=journal_counts.index, ax=ax1, palette="viridis", hue=journal_counts.index, legend=False)
    ax1.set_title(f'Distribution by Journal (Sample of {len(df_stats)} Papers)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('')
    # Add count labels
    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    # Plot 2: Curator Distribution 
    ax2 = fig5.add_subplot(gs5[1])
    curator_counts = df_stats['Curator'].value_counts()
    # Using a simple bar plot to show the count of papers per curator in this sample
    sns.barplot(x=curator_counts.values, y=curator_counts.index, ax=ax2, palette="magma", hue=curator_counts.index, legend=False)
    ax2.set_title(f'Distribution by Human Curator (Sample of {len(df_stats)} Papers)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Papers Annotated in this Set')
    ax2.set_ylabel('')
    # Add count labels
    for container in ax2.containers:
        ax2.bar_label(container, padding=3)

    plt.tight_layout()
    filepath_05 = os.path.join(PLOTS_DIR, '05_Diversity_Journal_Curator.png')
    plt.savefig(filepath_05)
    plt.close()
    print(f"Saved: {filepath_05}")


    # --- Plot 06: File Statistics (PDF Size + Supplementary) ---
    fig6 = plt.figure(figsize=(16, 8))
    gs6 = fig6.add_gridspec(1, 2)

    # Plot 3: PDF Size (Simple Histogram with Counts)
    ax3 = fig6.add_subplot(gs6[0])
    # Binning the sizes for better readability
    # Round sizes to nearest 0.5 MB for grouping? Or just standard histograms
    sns.histplot(data=df_stats, x='Main_PDF_Size_MB', bins=10, ax=ax3, kde=False, color='#2e86c1')
    ax3.set_title('Main PDF File Size Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Size (MB)')
    ax3.set_ylabel('Count of Papers')
    # Label counts on bars
    for container in ax3.containers:
        ax3.bar_label(container)

    # Plot 4: Supplementary Files Count (Simple Bar Dist)
    ax4 = fig6.add_subplot(gs6[1])
    
    # Get frequency of counts (e.g., how many papers have 0 supps, 1 supp, etc.)
    supp_dist = df_stats['Supplementary_Files_Count'].value_counts().sort_index()
    
    sns.barplot(x=supp_dist.index, y=supp_dist.values, ax=ax4, color='#2ca02c')
    ax4.set_title('Supplementary Files Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Supplementary Files per Paper')
    ax4.set_ylabel('Count of Papers')
    # Label counts
    for container in ax4.containers:
        ax4.bar_label(container)

    plt.tight_layout()
    filepath_06 = os.path.join(PLOTS_DIR, '06_File_Statistics.png')
    plt.savefig(filepath_06)
    plt.close()
    print(f"Saved: {filepath_06}")


    # Display Summary Stats
    print("\n=== Diversity Summary ===")
    print(f"Total Entries: {len(df_stats)}")
    print(f"Unique Journals: {df_stats['Journal'].nunique()}")
    print(f"Unique Curators: {df_stats['Curator'].nunique()}")
    print(f"Avg Main PDF Size: {df_stats['Main_PDF_Size_MB'].mean():.2f} MB")
    print(f"Avg Supplementary Files: {df_stats['Supplementary_Files_Count'].mean():.1f}")
    if 'Supplementary_Files_Count' in df_stats:
        print(f"Entries with 0 Supplementary Files: {len(df_stats[df_stats['Supplementary_Files_Count'] == 0])}")


def load_registry_mappings():
    print("Loading User and DOI Mappings from Registry...")
    doi_to_oid = {}
    oid_to_user = {}

    reviews_path = os.path.join(REGISTRY_DIR, RAW_REVIEWS_FILE)
    users_path = os.path.join(REGISTRY_DIR, USERS_FILE)

    if not os.path.exists(reviews_path) or not os.path.exists(users_path):
        print("Registry JSON files not found. Skipping mapping load.")
        return {}, {}

    try:
        with open(reviews_path, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
            for entry in reviews_data:
                doi = entry.get('publication', {}).get('doi', '')
                oid = entry.get('user', {}).get('$oid', '')
                if doi and oid:
                    doi_to_oid[doi.strip()] = oid
    except Exception as e:
        print(f"Error loading reviews: {e}")

    try:
        with open(users_path, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
            for u in users_data:
                oid = u.get('_id', {}).get('$oid', '')
                if oid:
                    name = f"{u.get('name', '')} {u.get('surname', '')}".strip()
                    if not name and u.get('email'):
                        name = u.get('email')
                    if not name:
                        name = "Unknown"
                    oid_to_user[oid] = name
    except Exception as e:
        print(f"Error loading users: {e}")
        
    return doi_to_oid, oid_to_user

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    setup_plots_dir()
    
    # Load Main Evaluation Results
    df = load_evaluation_results()
    
    if df is not None:
        plot_overall_rank_distribution(df)
        plot_win_rate_analysis(df)
        plot_performance_by_field(df)
        plot_copilot_win_rate_per_field(df)
        
        # Load mappings for diversity analysis
        doi_to_oid, oid_to_user = load_registry_mappings()
        
        # Run Diversity Analysis (if data folder found)
        run_diversity_analysis(doi_to_oid, oid_to_user)
        
    print(f"\nAnalysis Complete. Check '{os.path.basename(PLOTS_DIR)}' for outputs.")
