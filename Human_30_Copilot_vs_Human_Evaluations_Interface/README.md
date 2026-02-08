# Human Evaluation Interface (Copilot vs Human)

This directory contains the tools and data for the comparative analysis between DOME Copilot's automated curation and human expert curation.

## Contents

- **`evaluation_app.py`**: A Python application (likely Streamlit or similar dashboard) used to visualize and facilitate the human evaluation process.
- **`Analysis_Evaluation_Results.ipynb`**: A Jupyter Notebook for analyzing the results of the comparison, calculating agreement metrics (precision, recall, F1, Cohen's kappa, etc.).
- **`30_human_evaluation/`**: Directory containing the specific subset of 30 papers selected for detailed human review.
- **`evaluation_results.tsv`**: The recorded results of the evaluation session.

## Purpose

To validate the accuracy of the DOME Copilot, a subset of 30 papers was randomly selected and manually reviewed. This interface supports the workflow of comparing the AI-generated metadata against the ground truth established by human experts.

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

