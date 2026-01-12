#!/bin/bash

# Installation verification script
echo "=============================================="
echo "Speech-to-Text System - Installation Check"
echo "=============================================="
echo ""

# Check Python
echo "Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python not found"
    exit 1
fi

# Check virtual environment
echo ""
echo "Checking virtual environment..."
if [ -d "sst" ]; then
    echo "✓ Virtual environment exists: sst/"
    
    # Activate and check
    source sst/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found"
    echo "  Create it with: python3 -m venv sst"
    exit 1
fi

# Check ffmpeg
echo ""
echo "Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n 1)
    echo "✓ ffmpeg installed"
else
    echo "✗ ffmpeg not found"
    echo "  Install it with: brew install ffmpeg (macOS)"
    exit 1
fi

# Check Python packages
echo ""
echo "Checking Python packages..."

packages=(
    "torch"
    "torchaudio"
    "whisper"
    "numpy"
    "sklearn"
    "jiwer"
    "tqdm"
)

all_installed=true

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package"
    else
        echo "✗ $package not installed"
        all_installed=false
    fi
done

if [ "$all_installed" = false ]; then
    echo ""
    echo "Install missing packages with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check directory structure
echo ""
echo "Checking directory structure..."

dirs=(
    "src"
    "data"
    "models"
    "configs"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ not found"
        mkdir -p "$dir"
        echo "  Created $dir/"
    fi
done

# Check required files
echo ""
echo "Checking required files..."

files=(
    "prepare_dataset.py"
    "pipeline.py"
    "evaluate_model.py"
    "src/model.py"
    "src/train.py"
    "src/inference.py"
    "src/dataset.py"
    "src/utils.py"
    "configs/config.json"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file not found"
    fi
done

# GPU check
echo ""
echo "Checking GPU availability..."
GPU_AVAILABLE=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')")

if [ "$GPU_AVAILABLE" = "yes" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✓ GPU available: $GPU_NAME"
    echo "  Training will be accelerated!"
else
    echo "○ No GPU detected"
    echo "  Training will use CPU (slower)"
fi

# Summary
echo ""
echo "=============================================="
echo "Installation Check Complete!"
echo "=============================================="
echo ""

if [ "$all_installed" = true ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "You're ready to start!"
    echo ""
    echo "Next steps:"
    echo "1. Run demo guide:     python demo.py"
    echo "2. Add audio files to: data/audio/"
    echo "3. Run pipeline:       python pipeline.py --audio-dir data/audio"
    echo ""
else
    echo "✗ Some checks failed"
    echo "  Please fix the issues above before proceeding"
    exit 1
fi
