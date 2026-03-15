# Human Evaluation Interface (Copilot vs Human)

This directory contains the tools and data for the comparative analysis between DOME Copilot's automated curation and human expert curation.

## Contents

- **`evaluation_app.py`**: A Python application (Tkinter-based) used to visualize and facilitate the human evaluation process.
- **`generate_statistical_analysis.py`**: A robust statistical analysis script that processes `evaluation_results.tsv`. It performs T-tests and Binomial tests to validate the significance of the results and generates `statistical_analysis_report.txt`.
- **`generate_evaluation_analysis_plots.py`**: Generates a comprehensive suite of visualization plots (Diversity, Performance, Precision/Recall, etc.) saved to the `evaluation_analysis_plots/` directory. Handles anonymization of curator IDs to protect privacy.
- **`Analysis_Evaluation_Results.ipynb`**: (Legacy) A Jupyter Notebook originally used for analyzing the results.
- **`30_human_evaluation/`**: Directory containing the specific subset of 30 papers selected for detailed human review.
- **`evaluation_results.tsv`**: The recorded results of the evaluation session.
- **`statistical_analysis_report.txt`**: The output report containing p-values and statistical findings.

## Purpose

To validate the accuracy of the DOME Copilot, a subset of 30 papers was randomly selected and manually reviewed. This interface supports the workflow of comparing the AI-generated metadata against the ground truth established by human experts.

## Analysis & Visualization

In addition to the evaluation interface, this folder contains the analytical engine for the study:

1.  **Statistical Analysis**: Run `python generate_statistical_analysis.py` to calculate significance tests between Copilot and Human performance.
2.  **Visualization**: Run `python generate_evaluation_analysis_plots.py` to create the 6 key evaluation plots (saved to `evaluation_analysis_plots/`). These plots include:
    *   Agreement/Disagreement rates.
    *   Data Diversity (Categories/Tasks).
    *   Evaluation Time metrics.
    *   Curator Performance comparisons (Anonymized).

## Running with Docker

To run the evaluation application in a clean environment, you can use the provided Docker container. This ensures all dependencies (including Tkinter and PDF viewing capabilities) are correctly installed.

### Prerequisites

- Docker installed on your system.
- An X11 server (running via Linux desktop).

### Setup and Run

1.  **Navigate to this directory:**
    ```bash
    cd Human_30_Copilot_vs_Human_Evaluations_Interface
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t evaluation_app .
    ```

3.  **Allow Docker to access your local X display:**
    ```bash
    xhost +local:docker
    ```

4.  **Run the container:**
    ```bash
    docker run -it --rm \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$(pwd)":/app \
        evaluation_app
    ```

**Notes:**
- The `-v "$(pwd)":/app` flag mounts the current directory into the container. This allows the app to read your data files and **save changes** to `evaluation_results.tsv` directly on your host machine.
- The `xhost` command is required to allow the GUI application inside the container to display windows on your host screen. You can revert this permission with `xhost -local:docker` after you are done if you wish.
- PDF Opening: The container includes a PDF viewer (`evince`). When you click to open a PDF in the app, it should launch the viewer window on your screen.

