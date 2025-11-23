#!/bin/bash
# Consolidated push script
# Pushes main repo to GitHub and data folder to Hugging Face

set -e

echo "=== Pushing to GitHub (origin) ==="
git push origin main

echo ""
echo "=== Pushing data folder to Hugging Face ==="
cd data
git push hf main
cd ..

echo ""
echo "=== Push complete ==="
echo "GitHub: [Your GitHub repo URL]"
echo "HF Dataset: [Your Hugging Face dataset URL]"
