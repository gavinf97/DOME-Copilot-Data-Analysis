import re

with open('Human_30_Copilot_vs_Human_Evaluations_Interface/generate_evaluation_analysis_plots.py', 'r') as f:
    text = f.read()

# Change hspace from 0.3 to 0.4 or 0.5 in gridspec
pattern = r"fig5 = plt\.figure\(figsize=\(16, 12\)\) \n    gs5 = fig5\.add_gridspec\(2, 1, height_ratios=\[1, 1\], hspace=0\.3\)"
replacement = """fig5 = plt.figure(figsize=(16, 13)) 
    gs5 = fig5.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)"""

new_text = re.sub(pattern, replacement, text)

# Also increase top and bottom margins as needed to not cramp
pattern2 = r"plt\.subplots_adjust\(left=0\.25, right=0\.95, top=0\.92, bottom=0\.08\)"
replacement2 = """plt.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.08, hspace=0.45)"""
new_text = re.sub(pattern2, replacement2, new_text)

with open('Human_30_Copilot_vs_Human_Evaluations_Interface/generate_evaluation_analysis_plots.py', 'w') as f:
    f.write(new_text)

print("Updated spacing...")
