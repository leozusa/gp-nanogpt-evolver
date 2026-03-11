#!/bin/bash
set -e

echo "=== Installing nanoGPT Evolver v1.0 ==="

# Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# Install dependencies
echo "Installing Python dependencies..."
pip install torch pandas plotly streamlit

# Prepare data
echo "Preparing Shakespeare character dataset..."
cd nanoGPT
python data/shakespeare_char/prepare.py 2>/dev/null || true
cd ..

echo ""
echo "✅ INSTALLATION COMPLETE!"
echo "You can now run: ./main.sh start"
