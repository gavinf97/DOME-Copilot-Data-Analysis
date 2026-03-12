FROM r-base:4.3.1

RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir pandas nltk evaluate bert-score torch

RUN R -e "install.packages(c('data.table', 'stringr', 'ggplot2', 'ggtext', 'ggh4x', 'colorspace', 'ggforce'), repos='http://cran.rstudio.com/')"

WORKDIR /workspace
