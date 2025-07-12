# CHRA-NF-XL Docker Build Environment
# Base: Ubuntu 22.04 LTS
FROM ubuntu:22.04

# 1. Environment Setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG C.UTF-8
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="/opt/vcpkg/installed/x64-linux/tools/cmake:${PATH}"

# 2. Install Core Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    zip \
    unzip \
    pkg-config \
    software-properties-common \
    wget \
    cmake \
    g++-11 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# 3. Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh

# 4. Install Python Dependencies
RUN python3.10 -m pip install --no-cache-dir --upgrade pip \
    && python3.10 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    onnx \
    onnxruntime \
    numpy \
    pybind11

# 5. Set up Working Directory
WORKDIR /workspace

# 6. Default Command
CMD ["/bin/bash"]
