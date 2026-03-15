# AlphaFold2 Copilot v2 vs Human Evaluation

This folder contains a single-publication evaluation package for:

- PMCID: `PMC8371605`
- PMID: `34265844`
- DOI: `10.1038/s41586-021-03819-2`
- Title: `Highly accurate protein structure prediction with AlphaFold.`

## Goal

Replicate the Human vs Copilot evaluation workflow used for the 30-paper interface, but only for one publication.

## Included Interface Assets

Copied from `Human_30_Copilot_vs_Human_Evaluations_Interface/`:

- `evaluation_app.py`
- `Dockerfile`
- `README_source_interface.md` (reference copy of source README)

Note: no notebook file was present in the source interface folder to copy.

## Single-Paper Source Data Layout

Expected interface data layout is preserved under:

- `30_Evaluation_Source_JSONs_Human_and_Copilot_Including_PDFs/PMC8371605/`

Files included:

- `PMC8371605_human.json`
- `PMC8371605_copilot.json`
- `PMC8371605_main.pdf`
- `41586_2021_3819_MOESM1_ESM.pdf`
- `41586_2021_3819_MOESM2_ESM.pdf`

## Provenance of Each File

- Human JSON source record:
  - `DOME_Registry_Human_Reviews_258_20260205.json`
  - Extracted by matching `publication/pmcid == PMC8371605`
  - Flattened to key format used by interface (e.g. `dataset/provenance`, `model/output`)

- Copilot JSON source:
  - `Copilot_Processed_Datasets_JSON/Copilot_222_v2_Processed_2026-03-02_Updated_Metadata/PMC8371605.json`

- PDF and supplementary files source:
  - `DOME_Registry_222_PMCID_PDFs_PMC_Full_Text_and_Supplementary/PMC8371605/`

## Run

From repository root:

```bash
python AlphaFold2_Copilot_v2_vs_Human_Evaluation/evaluation_app.py
```

The app will discover only one PMCID folder (`PMC8371605`) in this package.
