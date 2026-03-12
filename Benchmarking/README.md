# Benchmarking

This directory contains scripts and files for evaluating the performance of DOME Copilot versus the DOME registry data.

## Contents
- **v1/**: Contains results from the first version of the evaluation.
- **v2/**: Contains results from the second version of the evaluation.
- **metrics.py**: Python script for calculating text-based metrics (BERTScore, etc.).
- **benchmarking_*.R**: R scripts to generate plots and statistics based on the metrics.
- **Dockerfile**: A Docker container to help reproduce the R scripts.

## Running the R Scripts via Docker
To run the R scripts without installing dependencies locally, you can use the provided Docker container. 

1. Build the image:
   ```bash
   cd Benchmarking
   docker build -t dome_benchmarking -f Dockerfile .
   ```

2. Run the image:
   ```bash
   docker run --rm -v $(pwd):/workspace dome_benchmarking Rscript benchmarking_panel.R
   ```

*Note: The images originally generated directly inside `Benchmarking/` have been moved to their respective `v1/` or `v2/` directories to prevent duplicates.*
