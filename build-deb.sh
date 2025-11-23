#!/bin/bash
# Build and install Whisper Fine-Tuning Data as a Debian package
# Usage: ./build-deb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Package info
PKG_NAME="whisper-finetuning-data"
VERSION=$(grep -m1 "^${PKG_NAME}" debian/changelog | sed -E 's/.*\(([^)]+)\).*/\1/' | cut -d'-' -f1)

echo "Building ${PKG_NAME} version ${VERSION}"

# Check for required build tools
if ! command -v dpkg-deb &> /dev/null; then
    echo "Error: dpkg-deb not found. Install with: sudo apt install dpkg-dev"
    exit 1
fi

# Create build directory
BUILD_DIR="${SCRIPT_DIR}/build/${PKG_NAME}_${VERSION}_all"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Create directory structure
mkdir -p "${BUILD_DIR}/DEBIAN"
mkdir -p "${BUILD_DIR}/usr/bin"
mkdir -p "${BUILD_DIR}/usr/share/${PKG_NAME}/app"
mkdir -p "${BUILD_DIR}/usr/share/applications"
mkdir -p "${BUILD_DIR}/usr/share/icons/hicolor/128x128/apps"
mkdir -p "${BUILD_DIR}/usr/share/doc/${PKG_NAME}"

# Copy application files
cp app/*.py "${BUILD_DIR}/usr/share/${PKG_NAME}/app/"
chmod 644 "${BUILD_DIR}/usr/share/${PKG_NAME}/app/"*.py

# Copy launcher script
cp debian/whisper-finetuning-data.sh "${BUILD_DIR}/usr/bin/whisper-finetuning-data"
chmod 755 "${BUILD_DIR}/usr/bin/whisper-finetuning-data"

# Copy desktop entry
cp debian/whisper-finetuning-data.desktop "${BUILD_DIR}/usr/share/applications/"

# Copy icon if exists
if [ -f debian/whisper-finetuning-data.png ]; then
    cp debian/whisper-finetuning-data.png "${BUILD_DIR}/usr/share/icons/hicolor/128x128/apps/"
fi

# Copy documentation
cp README.md "${BUILD_DIR}/usr/share/doc/${PKG_NAME}/" 2>/dev/null || echo "No README found"
cp debian/copyright "${BUILD_DIR}/usr/share/doc/${PKG_NAME}/"

# Create control file
cat > "${BUILD_DIR}/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: sound
Priority: optional
Architecture: all
Depends: python3, python3-pyqt6, python3-pyqt6.qtmultimedia, python3-numpy, python3-requests, python3-dotenv
Maintainer: Daniel Rosehill <public@danielrosehill.com>
Description: Whisper Fine-Tuning Data Collection Tool
 A PyQt6 desktop application for collecting speech-to-text training data.
 Records audio samples paired with text prompts to create datasets for
 fine-tuning Whisper and other speech recognition models.
EOF

# Create postinst script
cat > "${BUILD_DIR}/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

echo ""
echo "Whisper Fine-Tuning Data Collector installed successfully!"
echo ""
echo "To run: whisper-finetuning-data"
echo ""
echo "Data will be stored in: ~/.local/share/whisper-finetuning-data/"
echo "Add your OpenRouter API key to: ~/.local/share/whisper-finetuning-data/.env"
echo ""

exit 0
EOF
chmod 755 "${BUILD_DIR}/DEBIAN/postinst"

# Create postrm script
cat > "${BUILD_DIR}/DEBIAN/postrm" << 'EOF'
#!/bin/bash
set -e

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

exit 0
EOF
chmod 755 "${BUILD_DIR}/DEBIAN/postrm"

# Build the package
DEB_FILE="${SCRIPT_DIR}/build/${PKG_NAME}_${VERSION}_all.deb"
dpkg-deb --build --root-owner-group "${BUILD_DIR}" "${DEB_FILE}"

echo ""
echo "Package built: ${DEB_FILE}"
echo ""

# Offer to install
read -p "Install the package now? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Installing ${PKG_NAME}..."
    sudo dpkg -i "${DEB_FILE}"

    # Install dependencies if needed
    if [ $? -ne 0 ]; then
        echo "Installing dependencies..."
        sudo apt-get install -f -y
    fi

    echo ""
    echo "Installation complete!"
    echo ""
    echo "Run with: whisper-finetuning-data"
else
    echo "To install manually: sudo dpkg -i ${DEB_FILE}"
fi
