# Download Negative 1012 PMC Full Text and Supplementary

This directory contains the pipeline for downloading full-text PDFs and supplementary materials for the **1012 Negative Dataset**. These are articles that were identified as negative examples in the DOME analysis context.

## Scripts

- **`Download_1012_Negative_PMC_Full_Text_and_Supplementary.py`**: The main script that iterates through the input CSV, checks for PMCIDs, and downloads the corresponding files from PubMed Central/Europe PMC.

## Input Data

- **`negative_entries_pmid_pmcid_filtered.csv`**: A CSV file containing the list of articles (PMID/PMCID) to be processed. This file contains 1346 entries after filtering for valid PMCIDs.

## Output Directories

- **`Negative_PMC_PDFs/`**: Stores the downloaded full-text PDFs.
- **`Negative_PMC_Supplementary/`**: Stores downloaded supplementary material archives (and extracted files).
- **`Negative_PMC_TSV_Files/`**: Output metadata and tracking logs in TSV format.

## Usage

To run the download process in automated mode:

```bash
python Download_1012_Negative_PMC_Full_Text_and_Supplementary.py --automated
```

## Data Hosting

The resulting large binary datasets (PDFs and Supplementary files) are not tracked in this repository. They will be archived and hosted on **Zenodo** upon publication.
