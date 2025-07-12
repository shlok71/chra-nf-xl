# CHRA-NF-XL Windows Bootstrap Script (PowerShell)

# --- Configuration ---
$VcpkgUri = "https://github.com/microsoft/vcpkg.git"
$VcpkgDir = "$PSScriptRoot/../vcpkg"
$VcpkgCommit = "master" # Or a specific commit hash for reproducibility
$Libraries = @(
    "pybind11",
    "eigen3",
    "gtest"
)

# --- 1. Install vcpkg ---
if (-not (Test-Path -Path $VcpkgDir)) {
    Write-Host "Cloning vcpkg..."
    git clone $VcpkgUri --branch $VcpkgCommit --depth 1 $VcpkgDir
} else {
    Write-Host "vcpkg directory already exists."
}

if (-not (Test-Path -Path "$VcpkgDir/vcpkg.exe")) {
    Write-Host "Bootstrapping vcpkg..."
    & "$VcpkgDir/bootstrap-vcpkg.bat"
} else {
    Write-Host "vcpkg already bootstrapped."
}

# --- 2. Install Dependencies with vcpkg ---
Write-Host "Installing required libraries via vcpkg..."
$env:VCPKG_DEFAULT_TRIPLET = 'x64-windows'
& "$VcpkgDir/vcpkg.exe" install $Libraries --recurse

# --- 3. Install Python Dependencies ---
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu onnx onnxruntime numpy pybind11

# --- 4. Configure CMake & Build ---
$BuildDir = "$PSScriptRoot/../build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir
}

Write-Host "Configuring CMake..."
$ToolchainFile = "$VcpkgDir/scripts/buildsystems/vcpkg.cmake"
cmake -B $BuildDir -S "$PSScriptRoot/.." -DCMAKE_TOOLCHAIN_FILE=$ToolchainFile

Write-Host "Building project..."
cmake --build $BuildDir --config Release

Write-Host "Bootstrap complete. Build artifacts are in '$BuildDir'."
