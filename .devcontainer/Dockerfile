# Start from Rocker devcontainer base image
FROM ghcr.io/rocker-org/devcontainer/tidyverse:4

# Install additional R packages
RUN install2.r --error --skipinstalled --ncpus -2 \
    keras ggplot2 dplyr xgboost tidymodels ranger \
    && rm -rf /tmp/downloaded_packages \
    && strip /usr/local/lib/R/site-library/*/libs/*.so

# Install system dependencies for TensorFlow and GIS support
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libgeos-dev libudunits2-dev libproj-dev libgdal-dev libxt6 libglpk-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Download & compile Python 3.9 with --enable-shared
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz && \
    tar xzf Python-3.9.18.tgz && \
    cd Python-3.9.18 && \
    ./configure --enable-shared --prefix=/usr/local LDFLAGS="-Wl,-rpath /usr/local/lib" && \
    make -j$(nproc) && \
    make altinstall

# Ensure the new Python is the default
RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

# Create a virtual environment for TensorFlow
#RUN python3.9 -m venv /opt/tf-venv

# Activate the virtual environment
#ENV PATH="/opt/tf-venv/bin:$PATH"

# Upgrade pip and install required Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow tensorflow-hub tensorflow-datasets \
    scipy requests Pillow h5py pandas pydot

# Ensure R uses the correct Python
RUN echo "RETICULATE_PYTHON=/usr/local/bin/python3.9" >> /etc/R/Renviron.site

# Configure RStudio authentication
RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment
