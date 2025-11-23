#!/bin/bash
# Run the Whisper Fine-Tuning Data Collector

cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Use python3 explicitly (python may not be available on some systems)
python3 app/desktop.py
