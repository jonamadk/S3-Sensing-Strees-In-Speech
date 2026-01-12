#!/bin/bash

# Quick start script for Speech-to-Text Transformer project

echo "==================================="
echo "Speech-to-Text Transformer Setup"
echo "==================================="

# Check if virtual environment exists
if [ ! -d "sst" ]; then
    echo "Virtual environment not found. Please create it first:"
    echo "python3 -m venv sst"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source sst/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/audio
mkdir -p models/checkpoints
mkdir -p logs

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Place your audio files in data/audio/"
echo "2. Update data/train.json and data/val.json with your data"
echo "3. Adjust configs/config.json if needed"
echo "4. Run training: python src/train.py --config configs/config.json"
echo ""
echo "For more information, see README.md"
