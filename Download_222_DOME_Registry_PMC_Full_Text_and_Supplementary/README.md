# DOME Registry Downloader Script

This directory contains the Python script converted from the `Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary.ipynb` notebook.

## Usage

Run the script from inside this directory:

```bash
cd Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary
python3 Download_222_DOME_Registry_PMC_Full_Text_and_Supplementary.py
```

## Options

- `--automated`: Run all steps without stopping.
- `--skip-manual`: Skip manual remediation checks (useful for fully automated pipelines).
- `--force-download`: Force re-download of files even if they already exist (Default: skips existing files).

## Output Folders

The script will generate the following folders in your working directory:
- `DOME_Registry_JSON_Files/`
- `DOME_Registry_TSV_Files/`
- `DOME_Registry_PMC_PDFs/`
- `DOME_Registry_PMC_Supplementary/`
- `DOME_Registry_Remediation/`
