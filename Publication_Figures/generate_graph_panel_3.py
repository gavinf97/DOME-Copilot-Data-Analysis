#!/usr/bin/env python3
"""
generate_graph_panel_3.py

This script performs the analysis for Graph Panel 3, focusing on Information Coverage and Relevance
extracted by the Copilot across the 1012 dataset papers.

It generates three plots:
1. Coverage (Success Rate)
2. Relevance (Applicability)
3. Valid Information (Success AND Applicable)
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Source
# Adjusted path based on workspace structure
DATA_FOLDER = os.path.join(SCRIPT_DIR, "../Copilot_Processed_Datasets_JSON/Copilot_1012_v0_Processed_2026-01-15_Updated_Metadata")

# Output Directory
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Graph_Panel_3")

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def load_and_process_data():
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder not found at {DATA_FOLDER}")
        return None

    json_files = glob.glob(os.path.join(DATA_FOLDER, '*.json'))
    file_count = len(json_files)
    print(f"Found {file_count} JSON files in {DATA_FOLDER}")
    
    if file_count == 0:
        print("No files found. Exiting.")
        return None

    # Initialize counters
    target_missing = "not enough information"
    target_na = "not applicable"

    results = []

    categories_map = {
        'dataset': 'Data',
        'optimization': 'Optimisation',
        'model': 'Model',
        'evaluation': 'Evaluation'
    }

    print("Processing files...")
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for key, value in data.items():
                if '/' in key:
                    prefix, subfield = key.split('/', 1)
                    
                    if prefix in categories_map:
                        category = categories_map[prefix]
                        
                        is_missing = False
                        is_na = False
                        
                        if isinstance(value, str):
                            v_lower = value.lower()
                            if target_missing in v_lower:
                                is_missing = True
                            if target_na in v_lower:
                                is_na = True
                        
                        results.append({
                            'File': os.path.basename(json_file),
                            'Category': category,
                            'Subfield': subfield,
                            'IsMissing': is_missing,
                            'IsNA': is_na
                        })
                        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    df = pd.DataFrame(results)
    print(f"Processed {len(df)} records.")
    return df

# ==============================================================================
# PLOTTING
# ==============================================================================

def create_coverage_plot(data_df, condition_col, title_suffix, xlabel, filename, invert_condition=True):
    """
    Generates a coverage plot based on a specific boolean condition.
    invert_condition=True means we count where condition is FALSE (Inverse/Coverage).
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if invert_condition:
        subset_df = data_df[data_df[condition_col] == False]
    else:
        # Special case where a boolean Series/mask is passed directly
        if isinstance(condition_col, pd.Series):
             subset_df = data_df[condition_col]
        else:
             subset_df = data_df[data_df[condition_col] == True]

    counts = subset_df.groupby(['Category', 'Subfield']).size().reset_index(name='Count')
    
    total_files = data_df['File'].nunique()
    
    # Colors
    category_colors = {
        'Data': '#90C083', 
        'Optimisation': '#AEACDD', 
        'Model': '#7EB1DD', 
        'Evaluation': '#F8AEAE'
    }
    categories = ['Data', 'Optimisation', 'Model', 'Evaluation']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    # Create main figure title using the suffix
    fig.suptitle(title_suffix, fontsize=30, fontweight='bold')
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        subset = counts[counts['Category'] == category]
        
        # Merge with all subfields to ensure consistency (show 0 bars if missing)
        all_subfields = data_df[data_df['Category'] == category]['Subfield'].unique()
        full_subset = pd.DataFrame({'Subfield': all_subfields})
        full_subset = full_subset.merge(subset, on='Subfield', how='left').fillna(0)
        full_subset['Subfield'] = full_subset['Subfield'].str.capitalize()
        full_subset = full_subset.sort_values('Count', ascending=True)
        
        bar_color = category_colors.get(category, 'skyblue')
        bars = ax.barh(full_subset['Subfield'], full_subset['Count'], color=bar_color)
        
        # Simplify subplot title to just category, main title has the description
        ax.set_title(category, fontweight='bold', fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(0, 1200) # Increased limit slightly for larger labels
        
        for bar in bars:
            width = bar.get_width()
            # Bar count Text
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=16, fontweight='bold')
            # Percentage Text
            if width > 0:
                pct = (width / total_files) * 100
                ax.text(width / 2, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', 
                        ha='center', va='center', color='black', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    # plt.show() # Disabled for script execution
    plt.close()
    print(f"Graph saved to {path}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        print("Generating plots...")
        
        # --- 1. Plot 1: Not Enough Information (Prevalence of Missing Data) ---
        print("--- Plot 1: Not Enough Information ---")
        create_coverage_plot(
            df, 
            'IsMissing', 
            'Frequency of "Not enough information"', 
            'Number of Papers', 
            'graph_not_enough_info.png',
            invert_condition=False
        )

        # --- 2. Plot 2: Not Applicable (Prevalence of N/A) ---
        print("--- Plot 2: Not Applicable ---")
        create_coverage_plot(
            df, 
            'IsNA', 
            'Frequency of "Not applicable"', 
            'Number of Papers', 
            'graph_not_applicable.png',
            invert_condition=False
        )

        # --- 3. Plot 3: Joint/Yield (Neither Missing NOR NA) ---
        print("--- Plot 3: Joint Yield (Valid Information) ---")
        valid_mask = (df['IsMissing'] == False) & (df['IsNA'] == False)
        create_coverage_plot(
            df, 
            valid_mask, 
            'Yield: Useful Information (Extracted AND Applicable)', 
            'Number of Papers', 
            'graph_valid_yield.png',
            invert_condition=False 
        )
        
        print("Done.")
