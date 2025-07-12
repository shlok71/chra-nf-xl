# CI Smoke Test Script (Windows)

$ErrorActionPreference = "Stop"

Write-Host "Running smoke test..."

# In the future, this script would:
# 1. Run C++ tests via ctest.
# 2. Import Python bindings and run a simple function.
# 3. Execute a minimal inference task.

# For now, we just confirm the build directory exists and print success.
$BuildDir = "$PSScriptRoot/../build"
if (Test-Path -Path $BuildDir) {
    Write-Host "Build directory found."
    Write-Host "CI OK"
} else {
    Write-Host "Error: Build directory not found!"
    exit 1
}
