#!/bin/bash
# Launcher script for Whisper Fine-Tuning Data Collector

# Default data directory (can be overridden by WHISPER_FT_DATA_DIR env var)
if [ -z "${WHISPER_FT_DATA_DIR:-}" ]; then
    # Use XDG_DATA_HOME if set, otherwise use default
    DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/whisper-finetuning-data"
else
    DATA_DIR="${WHISPER_FT_DATA_DIR}"
fi

# Ensure directories exist
mkdir -p "${DATA_DIR}/audio" "${DATA_DIR}/text" "${DATA_DIR}/metadata"

# Copy .env template if not exists
if [ ! -f "${DATA_DIR}/.env" ]; then
    echo "# Add your OpenRouter API key here" > "${DATA_DIR}/.env"
    echo "OPENROUTER_API_KEY=" >> "${DATA_DIR}/.env"
fi

# Set environment to use the data directory
export WHISPER_FT_DATA_DIR="${DATA_DIR}"

# Run the application
cd /usr/share/whisper-finetuning-data
exec python3 app/desktop.py "$@"
