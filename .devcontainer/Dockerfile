FROM ghcr.io/rocker-org/devcontainer/tidyverse:4

## Adapted with thanks from github.com/louisaslett/MISCADA

RUN install2.r --error --skipinstalled --ncpus -2 \
    data.table \
    dplyr \
    forcats \
    foreach \
    ggplot2 \
    glmnet \
    jpeg \
    keras \
    keras3 \
    latex2exp \
    matrixStats \
    MASS \
    markdown \
    mgcv \
    mvtnorm \
    plotly \
    pracma \
    Polychrome \
    ranger \
    readr \
    rpart \
    stringr \
    tidyverse \
    xgboost \
    && rm -rf /tmp/downloaded_packages \
    && strip /usr/local/lib/R/site-library/*/libs/*.so

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    libgeos-dev \
    libudunits2-dev \
    libproj-dev \
    libgdal-dev \
    libxt6 \
    libglpk-dev \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages tensorflow \
    && pip install --break-system-packages tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot \
    && echo "RETICULATE_PYTHON=/usr/bin/python3" >> /usr/local/lib/R/etc/Renviron.site

RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment
