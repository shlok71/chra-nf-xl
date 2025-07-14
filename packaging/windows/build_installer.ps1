# This script builds a Windows installer for NeuroForge using PyInstaller.

# 1. Install PyInstaller
pip install pyinstaller

# 2. Bundle the application
pyinstaller --name "NeuroForge" `
    --onefile `
    --windowed `
    --add-data "module_registry.json;." `
    --add-data "user_config.json;." `
    --add-data "neuroforge_launcher.py;." `
    --add-data "bvh/libbvh.a;bvh" `
    --add-data "sgr/libsgr.a;sgr" `
    --add-data "reasoning/libreasoning.a;reasoning" `
    --add-data "nca/libnca.a;nca" `
    --add-data "memory/libmemory.a;memory" `
    --add-data "voice/libvoice.a;voice" `
    --add-data "overlay/liboverlay.a;overlay" `
    --add-data "quantization/libquantizer.a;quantization" `
    --add-data "scaling/libscaling.a;scaling" `
    --add-data "paging/libpaging.a;paging" `
    --add-data "profiling/libprofiler.a;profiling" `
    neuroforge_launcher.py

# 3. Create an installer
# This is a placeholder for creating an installer.
# A real implementation would use a tool like Inno Setup or NSIS.
New-Item -ItemType Directory -Force -Path "dist/NeuroForge"
Copy-Item -Path "dist/NeuroForge.exe" -Destination "dist/NeuroForge"
# Add other files to the installer here.
New-Item -ItemType Directory -Force -Path "dist/NeuroForge/data"
Copy-Item -Path "training/data" -Destination "dist/NeuroForge/data" -Recurse
# Create a dummy installer
New-Item -ItemType File -Force -Path "dist/NeuroForgeInstaller.exe"
Set-Content -Path "dist/NeuroForgeInstaller.exe" -Value "This is a dummy installer."
