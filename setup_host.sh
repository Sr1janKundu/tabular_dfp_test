#!/usr/bin/env bash

set -euxo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

REAL_USER=${SUDO_USER:-$USER}
UBUNTU_CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")

echo "Detected Ubuntu codename: ${UBUNTU_CODENAME}"

########################################
# Install Docker (if missing)
########################################

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker not found. Installing Docker..."

    apt update

    apt install -y \
        ca-certificates \
        curl \
        gnupg2

    install -m 0755 -d /etc/apt/keyrings

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc

    chmod a+r /etc/apt/keyrings/docker.asc

    tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

    apt update

    apt install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    systemctl enable docker
    systemctl restart docker

else
    echo "Docker already installed."
fi

########################################
# Install NVIDIA Container Toolkit
# (if missing)
########################################

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    echo "NVIDIA Container Toolkit not found. Installing..."

    apt update

    apt install -y \
        ca-certificates \
        curl \
        gnupg2

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L \
        https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt update

    apt install -y \
        nvidia-container-toolkit \
        nvidia-container-toolkit-base \
        libnvidia-container-tools \
        libnvidia-container1

else
    echo "NVIDIA Container Toolkit already installed."
fi

########################################
# Configure NVIDIA runtime
########################################

if ! grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
    echo "Configuring NVIDIA runtime for Docker..."

    nvidia-ctk runtime configure --runtime=docker

    systemctl restart docker
else
    echo "NVIDIA runtime already configured."
fi

########################################
# Add user to docker group
########################################

if ! getent group docker >/dev/null; then
    groupadd docker
fi

usermod -aG docker "$REAL_USER" || true

########################################
# Verification
########################################

echo
echo "Running GPU container verification..."

docker run --rm --gpus all \
    nvidia/cuda:12.8.0-devel-ubuntu24.04 \
    nvidia-smi

echo
echo "========================================"
echo "Setup complete."
echo "User added to docker group: $REAL_USER"
echo
echo "Restart shell / VM / WSL if needed."
echo "========================================"