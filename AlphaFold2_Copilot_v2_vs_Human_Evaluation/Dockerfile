FROM python:3.9-slim

# Install system dependencies
# python3-tk: for tkinter
# xdg-utils: for xdg-open
# evince: default pdf viewer (you can choose others like xpdf, but evince is standard)
# libx11-6: X11 library
RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    xdg-utils \
    evince \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install python dependencies
RUN pip install pandas

# The source code will be mounted at runtime to /app
# This ensures that generated files (evaluation_results.tsv) are saved to the host
# and we can access the data files.

CMD ["python", "evaluation_app.py"]
