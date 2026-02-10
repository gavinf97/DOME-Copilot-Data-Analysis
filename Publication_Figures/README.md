# Publication Figures and Analysis

This directory serves as the workspace for generating the final figures and analytical plots used in the associated research publication.

## Scripts & Notebooks

- **`generate_graph_panel_3.py`**: The primary Python script for generating **Panel 3** of the publication figures. It analyzes the "Coverage" and "Yield" of the Copilot extraction across the full 1012 dataset.
- **`Graph_Panel_3_Analysis.ipynb`**: (Legacy) The original notebook used for exploratory analysis of Panel 3.

## Outputs

Figures are generated into subdirectories corresponding to their panel number:

### **`Graph_Panel_3/`**
Contains the high-resolution (300 DPI) coverage analysis plots:
1.  **`graph_not_enough_info.png`**: Frequency of "Not enough information" responses (Mission Data).
2.  **`graph_not_applicable.png`**: Frequency of "Not applicable" responses (Field Relevance).
3.  **`graph_valid_yield.png`**: The joint yield showing fields that were both applicable and successfully extracted.

## Usage

To regenerate the figures:
```bash
python generate_graph_panel_3.py
```

This script reads from the `Copilot_Processed_Datasets_JSON/Copilot_1012_v0_Processed_2026-01-15_Updated_Metadata` directory.

