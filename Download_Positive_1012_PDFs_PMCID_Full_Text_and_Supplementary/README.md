# Download 1012 Positive PMC Full Text and Supplementary

This folder contains the automated Python script for downloading and processing the "Positive Set" of 1012 entries, converted from the `Download_1012_Positive_PMC_Full_Text_and_Supplementary.ipynb` notebook.

## Contents

- `Download_1012_Positive_PMC_Full_Text_and_Supplementary.py`: The main script.
- `Positive_PMC_PDFs/`: Directory where PDF full texts are downloaded.
- `Positive_PMC_Supplementary/`: Directory where supplementary files are extracted.
- `Positive_PMC_TSV_Files/`: Directory for status reports and analysis outputs.

## Usage

Run the script from within this directory or the root directory.

```bash
# From this directory
cd Download_1012_Positive_PMC_Full_Text_and_Supplementary
python3 Download_1012_Positive_PMC_Full_Text_and_Supplementary.py
```

### Options

- `--input-file <path>`: Specify a different input TSV file (default looks for `positive_entries_pmid_pmcid_filtered.tsv` in the current or parent directory).
- `--force-download`: Force re-download of files even if they already exist locally.

## Logic Overview

1.  **Load Data**: Reads the input TSV containing PMCIDs.
2.  **Download PDFs**: Retrieves full text PDFs from Europe PMC (Metadata or OA Render). Skips if already downloaded.
3.  **Download Supplementary**: Retrieves supplementary packages via NCBI OA XML API, extracting only PDF files. Skips if folder populated.
4.  **Organize**:
    *   Moves main PDFs into the supplementary folders.
    *   Removes exact duplicates.
    *   Cleans files based on negative keywords (e.g., "review", "revision").
5.  **Report**: Generates metrics and a chart (`Positive_Entries_Analysis_YYYY-MM-DD.png`) in the TSV folder.
