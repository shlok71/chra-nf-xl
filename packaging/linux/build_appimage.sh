#!/bin/bash

# This script builds a Linux AppImage for NeuroForge.

# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y fuse libfuse2

# 2. Download appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# 3. Create the AppDir
mkdir -p NeuroForge.AppDir/usr/bin
mkdir -p NeuroForge.AppDir/usr/lib

# 4. Copy the application and its dependencies
cp ../../neuroforge_launcher.py NeuroForge.AppDir/usr/bin/
cp ../../bvh/libbvh.so NeuroForge.AppDir/usr/lib/
# Copy other libraries here...

# 5. Create the AppRun script
cat > NeuroForge.AppDir/AppRun << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}/usr/lib"
exec "${HERE}/usr/bin/neuroforge_launcher.py" "$@"
EOF
chmod +x NeuroForge.AppDir/AppRun

# 6. Create the .desktop file
cat > NeuroForge.AppDir/neuroforge.desktop << 'EOF'
[Desktop Entry]
Name=NeuroForge
Exec=AppRun
Icon=neuroforge
Type=Application
Categories=Utility;
EOF

# 7. Create the icon
# This is a placeholder for the icon.
touch NeuroForge.AppDir/neuroforge.png

# 8. Build the AppImage
./appimagetool-x86_64.AppImage NeuroForge.AppDir
