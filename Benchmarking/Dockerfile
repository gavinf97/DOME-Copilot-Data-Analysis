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
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('data.table', 'stringr', 'ggplot2', 'ggtext', 'ggh4x', 'colorspace', 'ggforce'), repos='http://cran.rstudio.com/')"

WORKDIR /workspace
