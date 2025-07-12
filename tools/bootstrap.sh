#!/bin/bash
# CHRA-NF-XL Linux/macOS Bootstrap Script

set -e # Exit immediately if a command exits with a non-zero status.

# --- Detect OS ---
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=macOS;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${MACHINE}"

# --- 1. Install System Dependencies ---
echo "Installing system dependencies..."
if [ "${MACHINE}" == "Linux" ]; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git python3.10-dev pkg-config
elif [ "${MACHINE}" == "macOS" ]; then
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install it first."
        exit 1
    fi
    brew install cmake python@3.10 pkg-config
else
    echo "Unsupported OS: ${MACHINE}"
    exit 1
fi

# --- 2. Install Python Dependencies ---
echo "Installing Python dependencies..."
python3.10 -m pip install --upgrade pip
python3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cpu onnx onnxruntime numpy pybind11

# --- 3. Configure CMake & Build ---
BUILD_DIR="$(dirname "$0")/../build"
mkdir -p "${BUILD_DIR}"

echo "Configuring CMake..."
cmake -B "${BUILD_DIR}" -S "$(dirname "$0")/.."

echo "Building project..."
cmake --build "${BUILD_DIR}" --config Release -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "Bootstrap complete. Build artifacts are in '${BUILD_DIR}'."
