#!/bin/bash
# Backup/CLI sync helper for the Whisper Fine-Tuning dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="${SCRIPT_DIR}/data"

# Check for environment variable or use default
if [ -n "${WHISPER_FT_DATA_DIR:-}" ] && [ -d "${WHISPER_FT_DATA_DIR}/.git" ]; then
    DATA_DIR="$WHISPER_FT_DATA_DIR"
elif [ -d "${DEFAULT_DATA_DIR}/.git" ]; then
    DATA_DIR="$DEFAULT_DATA_DIR"
else
    echo "Unable to locate a git repository for the dataset."
    echo ""
    echo "Options:"
    echo "  1. Set WHISPER_FT_DATA_DIR=/path/to/your/dataset"
    echo "  2. Initialize git in ${DEFAULT_DATA_DIR}"
    echo "  3. Clone your Hugging Face dataset to ${DEFAULT_DATA_DIR}"
    echo ""
    echo "Example: git clone https://huggingface.co/datasets/your-username/your-dataset ${DEFAULT_DATA_DIR}"
    exit 1
fi

cd "$DATA_DIR"

echo "=== Whisper Fine-Tuning Dataset Backup Sync ==="
echo "Data directory: $DATA_DIR"
echo

if [ -d audio ]; then
    AUDIO_COUNT=$(find audio -maxdepth 1 -type f -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
else
    AUDIO_COUNT=0
fi
echo "Local audio files: $AUDIO_COUNT"
echo

STATUS=$(git status --porcelain)

if [ -z "$STATUS" ]; then
    echo "Working tree clean – nothing to sync."
    git fetch origin >/dev/null 2>&1 || true
    REMOTE_COUNT=$(git ls-tree -r origin/main --name-only audio/ 2>/dev/null | grep -c "\.wav$" || echo "0")
    echo "Remote audio files: $REMOTE_COUNT"
    exit 0
fi

echo "Changes to sync:"
echo "$STATUS"
echo

read -p "Commit and push the above changes? (y/N) " -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Sync cancelled."
    exit 0
fi

git add -A
COMMIT_MSG="Backup sync - $(date +'%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG"
git push

git fetch origin >/dev/null 2>&1 || true
REMOTE_COUNT=$(git ls-tree -r origin/main --name-only audio/ 2>/dev/null | grep -c "\.wav$" || echo "0")

echo "Sync complete."
echo "Local audio files: $AUDIO_COUNT"
echo "Remote audio files: $REMOTE_COUNT"
