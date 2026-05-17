FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# Base packages
RUN apt update && apt install -y \
    tzdata \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

########################################
# Immutable environment lives here
########################################

WORKDIR /opt/project

# Install Python
RUN uv python install 3.11.14

# Create uv project
RUN uv init

# Create venv
RUN uv venv --python 3.11.14

########################################
# Activate venv globally
########################################

ENV PATH="/opt/project/.venv/bin:${PATH}"

########################################
# Install packages
########################################

# Tensorboard
RUN uv add tensorboard

# RAPIDS first
RUN uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.4.*" \
    "dask-cudf-cu12==26.4.*" \
    "cuml-cu12==26.4.*" \
    "cugraph-cu12==26.4.*" \
    "nx-cugraph-cu12==26.4.*" \
    "cuxfilter-cu12==26.4.*" \
    "cucim-cu12==26.4.*" \
    "pylibraft-cu12==26.4.*" \
    "raft-dask-cu12==26.4.*" \
    "cuvs-cu12==26.4.*" \
    "nvforest-cu12==26.4.*"

# PyTorch after RAPIDS
RUN uv pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

########################################
# Workspace for mounted code
########################################

WORKDIR /workspace

CMD ["/bin/bash"]