Testing repo for NVIDIA Morpheus DFP pipeline for tabular data, without using Morpheus.

# Setup Dev Environment
> Tested with Ubuntu 24.04.4 LTS

# Install Docker Engine
```bash
# uninstall all conflicting packages
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)

# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# update system
sudo apt update

# install latest
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# verify that Docker is running
sudo systemctl status docker

# If Docker is not running, start it manually:
sudo systemctl start docker

# Verify that the installation is successful by running the hello-world image:
# sudo docker run hello-world

# Uninstall Docker Engine
# Uninstall the Docker Engine, CLI, containerd, and Docker Compose packages:
# sudo apt purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
# Images, containers, volumes, or custom configuration files on your host aren't automatically removed. To delete all images, containers, and volumes:
# sudo rm -rf /var/lib/docker
# sudo rm -rf /var/lib/containerd
# Remove source list and keyrings
# sudo rm /etc/apt/sources.list.d/docker.sources
# sudo rm /etc/apt/keyrings/docker.asc
```

## Install NVIDIA Container toolkit for Docker
```bash
# follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Install the prerequisites for the instructions below
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   ca-certificates \
   curl \
   gnupg2

# Configure the production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository
sudo apt-get update

# Install the NVIDIA Container Toolkit packages
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the container runtime by using the nvidia-ctk command
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon
sudo systemctl restart docker
```

## Install cuda 12.8
```bash
# follow https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# after installation, add to path
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

## Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or if system does not have curl: wget -qO- https://astral.sh/uv/install.sh | sh
```

## Setup UV venv and activate

```bash
uv python install 3.11.14
uv init
uv venv --python 3.11.14
source .venv/bin/activate

# install tensorboard & others before
uv add tensorboard sentence-transformers pyyaml notebook jupyterlab ipython ipykernel ipywidgets

# install rapids
uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.4.*" "dask-cudf-cu12==26.4.*" "cuml-cu12==26.4.*" \
    "cugraph-cu12==26.4.*" "nx-cugraph-cu12==26.4.*" "cuxfilter-cu12==26.4.*" \
    "cucim-cu12==26.4.*" "pylibraft-cu12==26.4.*" "raft-dask-cu12==26.4.*" \
    "cuvs-cu12==26.4.*" "nvforest-cu12==26.4.*" "nx-cugraph-cu12==26.4.*"

# install torch
uv install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# do it in this order: rapids -> torch -> other packages; this would mess up cuda dependency for rapids

# **do NOT do** something like the following, which would update pandas version and potentially others, resulting rapids not working due to dependency conflicts
# uv add jupyterlab ipython ipykernel ipywidgets
```
---
# With Docker 
## Prep host machine with
```bash
chmod +x setup_host.sh
sudo ./setup_host.sh
# chmod +x bootstrap_project.sh
# ./bootstrap_project.sh
```



## Build
```bash
docker build -t dfp_dev .
```

## Run
```bash
docker run --gpus all --network host -it \
    -v "$(pwd)":/workspace \
    dfp_dev
```

## resume the old container and interact with its shell exactly where you left off
```bash
docker start -ai <old_container_id_from_docker_ps_-a>
```

> Ignore everything, follow instructions.md