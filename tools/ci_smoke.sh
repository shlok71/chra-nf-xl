#!/bin/bash
# CI Smoke Test Script (Linux/macOS)

set -e

echo "Running smoke test..."
# In the future, this script would:
# 1. Run C++ tests via ctest.
# 2. Import Python bindings and run a simple function.
# 3. Execute a minimal inference task.

# For now, we just confirm the build directory exists and print success.
if [ -d "../build" ]; then
    echo "Build directory found."
    echo "CI OK"
else
    echo "Error: Build directory not found!"
    exit 1
fi
