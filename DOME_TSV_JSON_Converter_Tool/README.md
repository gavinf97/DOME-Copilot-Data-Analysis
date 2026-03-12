# DOME Registry Data Converter

This utility provides a robust, bidirectional conversion tool for DOME Registry data, allowing seamless transformation between flat TSV (Tab-Separated Values) files and nested JSON structures.

It uses local reference schema files to ensure that the document structure and column ordering remain consistent during conversion, preserving data integrity for the DOME Registry schema (including MongoDB-specific fields like `$oid` and `$date`).

## Features

- **TSV to JSON**: Converts flattened TSV spreadsheets back into the complex hierarchical JSON format required by the DOME application.
- **JSON to TSV**: Flattens nested JSON data into a readable TSV format suitable for analysis in Python/Pandas or Excel.
- **Reference-Driven**: Uses `DOME_Registry_Schema_Reference.json` and `DOME_Registry_Schema_Reference.tsv` to enforce correct field ordering and structure.

## Requirements

- Python 3.x
- `pandas` library

```bash
pip install pandas
```

## Usage

The script automatically detects the input file format based on the file extension (`.tsv` or `.json`) and performs the appropriate conversion.

### Convert TSV to JSON

```bash
python Convert_Registry_TSV_to_JSON.py your_data_file.tsv
```
*Output*: `your_data_file.json`

### Convert JSON to TSV

```bash
python Convert_Registry_TSV_to_JSON.py your_data_file.json
```
*Output*: `your_data_file.tsv`

## File Structure

- **`Convert_Registry_TSV_to_JSON.py`**: The main converter script.
- **`DOME_Registry_Schema_Reference.json`**: A sample JSON file (containing 2 records) that defines the correct nested structure.
- **`DOME_Registry_Schema_Reference.tsv`**: A header-only TSV file that defines the correct column order for flattened data.

> **Note**: These reference files must be present in the same directory as the script.

## Docker Setup

A Dockerfile is included in this directory to run the scripts in a containerized environment. Build the image with:
`docker build -t image-name .`
Run it with:
`docker run -it image-name`
