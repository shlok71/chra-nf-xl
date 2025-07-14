#!/bin/bash

# This script builds a macOS App for NeuroForge.

# 1. Create the .app directory structure
mkdir -p NeuroForge.app/Contents/MacOS
mkdir -p NeuroForge.app/Contents/Resources

# 2. Copy the executable
cp ../../neuroforge_launcher.py NeuroForge.app/Contents/MacOS/NeuroForge

# 3. Create the Info.plist file
cat > NeuroForge.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>NeuroForge</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.neuroforge</string>
</dict>
</plist>
EOF

# 4. Create the icon
# This is a placeholder for the icon.
touch NeuroForge.app/Contents/Resources/icon.icns

# 5. Create a DMG
hdiutil create -volname "NeuroForge" -srcfolder "NeuroForge.app" -ov -format UDZO "NeuroForge.dmg"
