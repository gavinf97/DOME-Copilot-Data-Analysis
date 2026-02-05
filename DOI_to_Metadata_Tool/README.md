# DOI to Metadata Converter

This versatile tool takes a Digital Object Identifier (DOI) and retrieves standardized publication metadata by intelligently querying a sequence of academic databases. It is designed to handle a wide range of academic outputs, from peer-reviewed journal articles to preprints and datasets.

## Key Features

- **Multi-Source Support:** Automatically falls back through multiple metadata providers to ensure the best possible result:
  1. **CrossRef:** The primary source for most academic publications.
  2. **Zenodo:** Dedicated support for datasets and software, including direct record lookups.
  3. **arXiv:** Specialized handling for physics, math, and cs preprints, including extracting arXiv IDs from DOIs.
  4. **BioRxiv & MedRxiv:** Specific API integration for detailed biology and medical preprints (distinguishing them from generic CSHL records).
  5. **Europe PMC:** Fallback using PMID for biomedical literature.

- **Smart ID Handling:** 
  - Resolves DOIs to PMIDs and PMCIDs using the NCBI Converter API.
  - Handles messy inputs (e.g., full URLs, dirtied strings) using robust regex cleaning.
  
- **Standardized Output:** Produces a consistent JSON format regardless of the source.

## Usage

1. **Install Dependencies:**
   Ensure you have Python 3 installed and the `requests` library:
   ```bash
   pip install requests
   ```

2. **Run the Script:**
   Provide the DOI (or a string containing it) as a command-line argument:
   ```bash
   python DOI_EPMC_Metadata_to_JSON.py "10.1038/s41586-020-2649-2"
   ```

   It handles messy inputs (URLs, text blocks) automatically:
   ```bash
   python DOI_EPMC_Metadata_to_JSON.py "Here is the paper: https://doi.org/10.1038/s41586-020-2649-2"
   ```
   
   It also supports Zenodo or arXiv DOIs directly:
   ```bash
   python DOI_EPMC_Metadata_to_JSON.py "10.5281/zenodo.16266118"
   ```

## Output

The tool prints the metadata to the console and saves a JSON file in the same directory.
- If a PMID is found: `metadata_32939066.json`
- If no PMID (e.g., datasets/preprints): `metadata_doi_10.5281_zenodo.16266118.json`

### Standardized Metadata Format
```json
{
    "publication/title": "Title of the paper or dataset",
    "publication/authors": "Author One, Author Two",
    "publication/journal": "Journal Name or 'Zenodo' / 'arXiv (Preprint)'",
    "publication/year": "2023",
    "publication/pmid": "12345678",   // Empty if not applicable
    "publication/pmcid": "PMC1234567", // Empty if not applicable
    "publication/doi": "10.1000/xyz"
}
```
