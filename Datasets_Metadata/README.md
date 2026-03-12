# Metadata

This folder contains scripts and output files related to collecting and managing metadata for the PMCIDs involved in the DOME Copilot Data Analysis datasets.

## Files

### Scripts
- **`fetch_dataset_metadata.py`**: The primary Python script used to consolidate and fetch metadata. It operates by reading the `dataset_pmcids.csv` file, separating PMCIDs into their respective dataset groups, and extracting metadata (`title`, `authors`, `journal`, `year`, `pmid`, `pmcid`, `doi`). 
  - **APIs used**: It first queries the Europe PMC API with batch processing (50 at a time). If some records are returned with missing fields, it falls back to the NCBI PubMed Central E-utilities API to try and fill in the gaps.
  - **Resiliency**: The script checks existing `.tsv` files and will skip records that are completely filled, only hitting the APIs for brand new PMCIDs or records that are currently missing columns.

### Inputs
- **`dataset_pmcids.csv`**: The central configuration list containing two columns: `pmcid` and `dataset`. To add new items to the metadata pull, append them to this CSV and run the Python script.

### Outputs
- **`DOME_Registry_222_Metadata.tsv`**
- **`Positive_1012_Metadata.tsv`**
- **`Negative_1012_Metadata.tsv`**
  - These TSV files contain the final consolidated metadata pulled for each respective dataset.
- **`coverage_report.txt`**: A dynamically generated text report showing the percentage of missing values across the datasets, along with the precise PMCIDs that failed to return certain fields (even after falling back to the NCBI API).

### Zenodo Dataset Information
- **`DOME_Copilot_Zenodo_Dataset_Package_Metadata.csv`**: This file acts as a data dictionary and manifest for the zipped dataset hosted on Zenodo. It provides qualitative descriptions for all the files and directories included in the Zenodo deposit, detailing what they are, where they belong within this repository's structure, and how users downloading the dataset can utilize them for reproduction or external analysis.

## Usage
To fetch or update metadata:
```bash
cd Metadata
python3 fetch_dataset_metadata.py
```
## Docker Setup

A Dockerfile is included in this directory to run the scripts in a containerized environment. Build the image with:
`docker build -t image-name .`
Run it with:
`docker run -it image-name`
