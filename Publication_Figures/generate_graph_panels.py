#!/usr/bin/env python3
"""
generate_graph_panels.py

This script performs the analysis for Graph Panels, focusing on Information Coverage and Relevance extracted by the Copilot across the dataset papers.
It handles nuanced parsing of multi-stacked responses to separate completely missing data from partial/mixed valid data.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Source
DATA_FOLDER = os.path.join(SCRIPT_DIR, "../Copilot_Processed_Datasets_JSON/Copilot_1012_v2_Pos_Processed_2026-03-02")
# Output Directory
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Graph_Panel_V2")

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def determine_yield_category(value_str):
    """
    Evaluates whether a multi-stacked question response is Fully Valid, Partially Valid,
    Fully Missing, or Fully NA.
    """
    v_lower = value_str.lower()
    
    # Strip basic layout headers to focus on answer content
    cleaned = re.sub(r'^\s*[\*-]?\s*\*\*?[^*:]+\*\*?\s*:?', '', v_lower, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*[\*-]\s+[^*:]+:\s?', '', cleaned, flags=re.MULTILINE)
    
    # Remove standard placeholder terms
    for term in ['not enough information is available.', 'not enough information.', 'not enough information', 'not applicable.', 'not applicable']:
        cleaned = cleaned.replace(term, '')
        
    chars_left = cleaned.translate(str.maketrans('','', string.punctuation)).replace(' ', '').replace('\n', '')
    has_valid_text = len(chars_left) > 0
    has_missing_str = "not enough information" in v_lower
    has_na_str = "not applicable" in v_lower
    
    if not has_valid_text:
        if has_missing_str:
            return 'Missing'
        if has_na_str:
            return 'NA'
        return 'Empty'
    else:
        if has_missing_str or has_na_str:
            return 'Partial'
        else:
            return 'Full'

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
                        
                        cat_res = 'Empty'
                        if isinstance(value, str):
                            cat_res = determine_yield_category(value)
                        
                        results.append({
                            'File': os.path.basename(json_file),
                            'Category': category,
                            'Subfield': subfield,
                            'IsMissing': cat_res == 'Missing',
                            'IsNA': cat_res == 'NA',
                            'IsPartial': cat_res == 'Partial',
                            'IsFull': cat_res == 'Full',
                            'IsValid': cat_res in ['Full', 'Partial']
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
    invert_condition=True means we count where condition is FALSE.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if invert_condition:
        subset_df = data_df[data_df[condition_col] == False]
    else:
        if isinstance(condition_col, pd.Series):
             subset_df = data_df[condition_col]
        else:
             subset_df = data_df[data_df[condition_col] == True]

    counts = subset_df.groupby(['Category', 'Subfield']).size().reset_index(name='Count')
    total_files = data_df['File'].nunique()
    
    category_colors = {
        'Data': '#90C083', 
        'Optimisation': '#AEACDD', 
        'Model': '#7EB1DD', 
        'Evaluation': '#F8AEAE'
    }
    categories = ['Data', 'Optimisation', 'Model', 'Evaluation']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight='bold')
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        subset = counts[counts['Category'] == category]
        
        all_subfields = data_df[data_df['Category'] == category]['Subfield'].unique()
        full_subset = pd.DataFrame({'Subfield': all_subfields})
        full_subset = full_subset.merge(subset, on='Subfield', how='left').fillna(0)
        full_subset['Subfield'] = full_subset['Subfield'].str.capitalize()
        full_subset = full_subset.sort_values('Count', ascending=True)
        
        bar_color = category_colors.get(category, 'skyblue')
        bars = ax.barh(full_subset['Subfield'], full_subset['Count'], color=bar_color)
        
        ax.set_title(category, fontweight='bold', fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 50)) 
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center', fontsize=16, fontweight='bold')
            if width > 0:
                pct = (width / total_files) * 100
                ax.text(width / 2, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%',
                        ha='center', va='center', color='black', fontsize=14, fontweight='bold')
                        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {path}")

# ==============================================================================
# MAIN
# ==============================================================================

def create_joint_stacked_plot(data_df, title_suffix, xlabel, filename):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    full_counts = data_df[data_df['IsFull'] == True].groupby(['Category', 'Subfield']).size().reset_index(name='FullCount')
    partial_counts = data_df[data_df['IsPartial'] == True].groupby(['Category', 'Subfield']).size().reset_index(name='PartialCount')
    
    total_files = data_df['File'].nunique()
    
    categories = ['Data', 'Optimisation', 'Model', 'Evaluation']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight='bold')
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        
        all_subfields = data_df[data_df['Category'] == category]['Subfield'].unique()
        df_sub = pd.DataFrame({'Subfield': all_subfields})
        
        f_sub = full_counts[full_counts['Category'] == category]
        p_sub = partial_counts[partial_counts['Category'] == category]
        
        df_sub = df_sub.merge(f_sub[['Subfield', 'FullCount']], on='Subfield', how='left').fillna(0)
        df_sub = df_sub.merge(p_sub[['Subfield', 'PartialCount']], on='Subfield', how='left').fillna(0)
        
        df_sub['TotalCount'] = df_sub['FullCount'] + df_sub['PartialCount']
        df_sub['Subfield'] = df_sub['Subfield'].str.capitalize()
        df_sub = df_sub.sort_values('TotalCount', ascending=True)
        
        # Plot Full
        bar1 = ax.barh(df_sub['Subfield'], df_sub['FullCount'], color='#27AE60', label='Full Yield')
        # Plot Partial on top
        bar2 = ax.barh(df_sub['Subfield'], df_sub['PartialCount'], left=df_sub['FullCount'], color='#2980B9', label='Partial Yield')
        
        ax.set_title(category, fontweight='bold', fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 150)) 

        # Add text
        for idx, row in df_sub.reset_index(drop=True).iterrows():
            total = row['TotalCount']
            full = row['FullCount']
            partial = row['PartialCount']
            subfield = row['Subfield'].lower()
            # Reduce inner bar font size only for evaluation/availability; total cap always full size
            fsize_inner = 15 if (category == 'Evaluation' and subfield == 'availability') else 20
            fsize_total = 20
            
            # 1) Total cap on end of bar in Black
            if total > 0:
                ax.text(total + 25, idx, f'{int(total)}', ha='left', va='center', fontsize=fsize_total, fontweight='bold', color='black')
                
            # 2) Full Yield subtotal completely inside or pointing inside, coloured White
            if full > 0:
                if full > 45 or total < 45:
                    ax.text(full / 2, idx, f'{int(full)}', ha='center', va='center', color='white', fontsize=fsize_inner, fontweight='bold')
                else:
                    # Move text horizontally into the partial bar's space, but keep it white
                    ax.annotate(f'{int(full)}', xy=(full / 2, idx), xytext=(full + 15, idx),
                                ha='left', va='center', color='white', fontsize=fsize_inner, fontweight='bold',
                                arrowprops=dict(arrowstyle="-", color='white', shrinkA=0, shrinkB=0, lw=1.5))
                                
            # 3) Partial Yield subtotal completely inside or pointing inside, coloured White
            if partial > 0:
                if partial > 45 or total < 45:
                    ax.text(full + (partial / 2), idx, f'{int(partial)}', ha='center', va='center', color='white', fontsize=fsize_inner, fontweight='bold')
                else:
                    # Move text horizontally into the full bar's space, keeping it white
                    ax.annotate(f'{int(partial)}', xy=(full + (partial / 2), idx), xytext=(full - 15, idx),
                                ha='right', va='center', color='white', fontsize=fsize_inner, fontweight='bold',
                                arrowprops=dict(arrowstyle="-", color='white', shrinkA=0, shrinkB=0, lw=1.5))

    # Single figure-level legend at top right
    import matplotlib.patches as mpatches
    full_patch = mpatches.Patch(color='#27AE60', label='Full Yield')
    partial_patch = mpatches.Patch(color='#2980B9', label='Partial Yield')
    total_patch = mpatches.Patch(color='black', label='Total (end of bar)')
    fig.legend(handles=[full_patch, partial_patch, total_patch],
               loc='upper right', fontsize=16, bbox_to_anchor=(0.99, 0.99),
               framealpha=0.9, edgecolor='black')
    
    # Adjusted tight_layout for more vertical space and top legend room
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=4.0)
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {path}")

def create_joint_grouped_plot(data_df, title_suffix, xlabel, filename):
    import numpy as np
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    full_counts = data_df[data_df['IsFull'] == True].groupby(['Category', 'Subfield']).size().reset_index(name='FullCount')
    partial_counts = data_df[data_df['IsPartial'] == True].groupby(['Category', 'Subfield']).size().reset_index(name='PartialCount')
    
    total_files = data_df['File'].nunique()
    
    categories = ['Data', 'Optimisation', 'Model', 'Evaluation']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
    fig.suptitle(title_suffix, fontsize=30, fontweight='bold')
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        
        all_subfields = data_df[data_df['Category'] == category]['Subfield'].unique()
        df_sub = pd.DataFrame({'Subfield': all_subfields})
        
        f_sub = full_counts[full_counts['Category'] == category]
        p_sub = partial_counts[partial_counts['Category'] == category]
        
        df_sub = df_sub.merge(f_sub[['Subfield', 'FullCount']], on='Subfield', how='left').fillna(0)
        df_sub = df_sub.merge(p_sub[['Subfield', 'PartialCount']], on='Subfield', how='left').fillna(0)
        
        df_sub['TotalCount'] = df_sub['FullCount'] + df_sub['PartialCount']
        df_sub['Subfield'] = df_sub['Subfield'].str.capitalize()
        df_sub = df_sub.sort_values('TotalCount', ascending=True)
        
        y = np.arange(len(df_sub))
        height = 0.35
        
        bar1 = ax.barh(y - height/2, df_sub['FullCount'], height, color='#27AE60', label='Full Yield')
        bar2 = ax.barh(y + height/2, df_sub['PartialCount'], height, color='#2980B9', label='Partial Yield')
        
        ax.set_yticks(y)
        ax.set_yticklabels(df_sub['Subfield'])
        
        ax.set_title(category, fontweight='bold', fontsize=24)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(0, max(1200, total_files + 50)) 
        
        if i == 0:
            ax.legend(fontsize=16)

        # Add text
        for b1, b2, idx in zip(bar1, bar2, range(len(df_sub))):
            w1 = b1.get_width()
            w2 = b2.get_width()
            if w1 > 0:
                ax.text(w1 + 10, b1.get_y() + b1.get_height()/2, f'{int(w1)}', ha='left', va='center', fontsize=12, fontweight='bold')
            if w2 > 0:
                ax.text(w2 + 10, b2.get_y() + b2.get_height()/2, f'{int(w2)}', ha='left', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved to {path}")


if __name__ == "__main__":
    df = load_and_process_data()
    
    if df is not None and not df.empty:
        print("Generating plots...")
        
        # 1. Entirely Missing Data
        print("--- Plot 1: Completely Missing (Not Enough Information) ---")
        create_coverage_plot(
            df, 
            'IsMissing', 
            'Frequency of Completely Missing Information', 
            'Number of Papers', 
            'graph_completely_missing.png',
            invert_condition=False
        )

        # 2. Not Applicable
        print("--- Plot 2: Not Applicable ---")
        create_coverage_plot(
            df, 
            'IsNA', 
            'Frequency of "Not applicable"', 
            'Number of Papers', 
            'graph_not_applicable.png',
            invert_condition=False
        )
        
        # 3. Partial Yield (Nuance for Multi-Stacked Qs)
        print("--- Plot 3: Partial Yield (Mixed Responses) ---")
        create_coverage_plot(
            df, 
            'IsPartial', 
            'Yield: Partial Information (Mixed Success on Sub-Qs)', 
            'Number of Papers', 
            'graph_partial_yield.png',
            invert_condition=False
        )

        # 4. Joint/Yield (Valid Information, Full + Partial)
        print("--- Plot 4: Overall Valid Yield (Full + Partial) ---")
        create_coverage_plot(
            df, 
            'IsValid', 
            'Overall Yield: Useful Information Extracted (Full & Partial)', 
            'Number of Papers', 
            'graph_valid_overall_yield.png',
            invert_condition=False 
        )
        
        # 5. Joint Stacked Plot (Full vs Partial in one bar)
        print("--- Plot 5: Joint Stacked Yield ---")
        create_joint_stacked_plot(
            df,
            'Joint Yield: Full vs Partial (Stacked)',
            'Number of Papers',
            'graph_joint_stacked_yield.png'
        )

        # 6. Joint Grouped Plot (Full vs Partial side-by-side)
        print("--- Plot 6: Joint Grouped Yield ---")
        create_joint_grouped_plot(
            df,
            'Joint Yield: Full vs Partial (Grouped)',
            'Number of Papers',
            'graph_joint_grouped_yield.png'
        )
        
        print("Done.")
