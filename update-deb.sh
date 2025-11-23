#!/bin/bash
# Update Whisper Fine-Tuning Data - increment version, build, and install
# Usage: ./update-deb.sh [major|minor|patch]
#        Default: patch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PKG_NAME="whisper-finetuning-data"
CHANGELOG="debian/changelog"

# Get current version from changelog
CURRENT_VERSION=$(grep -m1 "^${PKG_NAME}" "$CHANGELOG" | sed -E 's/.*\(([^)]+)\).*/\1/' | cut -d'-' -f1)
echo "Current version: ${CURRENT_VERSION}"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Determine increment type
INCREMENT_TYPE="${1:-patch}"

case "$INCREMENT_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo "Usage: $0 [major|minor|patch]"
        exit 1
        ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo "New version: ${NEW_VERSION}"

# Generate timestamp
TIMESTAMP=$(date -R)

# Prompt for changelog entry
echo ""
echo "Enter changelog entry (or press Enter for default):"
read -r CHANGELOG_ENTRY

if [ -z "$CHANGELOG_ENTRY" ]; then
    CHANGELOG_ENTRY="Version bump to ${NEW_VERSION}"
fi

# Create new changelog entry
NEW_ENTRY="${PKG_NAME} (${NEW_VERSION}-1) unstable; urgency=medium

  * ${CHANGELOG_ENTRY}

 -- Daniel Rosehill <public@danielrosehill.com>  ${TIMESTAMP}
"

# Prepend to changelog
echo "$NEW_ENTRY" | cat - "$CHANGELOG" > temp_changelog && mv temp_changelog "$CHANGELOG"

echo ""
echo "Updated changelog with version ${NEW_VERSION}"
echo ""

# Build the package
echo "Building package..."
./build-deb.sh

echo ""
echo "Update complete! Version ${NEW_VERSION} has been built and installed."
