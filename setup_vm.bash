#!/usr/bin/env bash

set -euxo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive

REAL_USER=${SUDO_USER:-$USER}
UBUNTU_CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")

########################################
# Docker
########################################

if ! command -v docker >/dev/null 2>&1; then

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
fi

########################################
# NVIDIA Container Toolkit
########################################

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then

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
fi

########################################
# Configure Docker runtime
########################################

if ! grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

########################################
# Docker group
########################################

if ! getent group docker >/dev/null; then
    groupadd docker
fi

usermod -aG docker "$REAL_USER" || true

########################################
# Verification
########################################

docker run --rm --gpus all \
    nvidia/cuda:12.8.0-devel-ubuntu24.04 \
    nvidia-smi