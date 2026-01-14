# Prosody Pipeline - Quick Start Guide

## 🎯 What This Pipeline Does

Extracts **phoneme-level prosodic features** (pitch, duration, energy) aligned with text transcriptions and uses them to detect **word stress patterns** using clustering or attention-based models.

## 📊 Output Example

```json
[
  {"phoneme": "IH", "start": 0.942, "end": 1.163, "word": "It", "pitch": 147.1, "energy": 0.023},
  {"phoneme": "T", "start": 1.163, "end": 1.184, "word": "It", "pitch": 158.15, "energy": 0.019},
  {"phoneme": " ", "start": 1.184, "end": 1.325, "word": " ", "pitch": 159.2, "energy": 0.015},
  {"phoneme": "IH", "start": 1.325, "end": 1.345, "word": "is", "pitch": 166.25, "energy": 0.028},
  {"phoneme": "Z", "start": 1.345, "end": 1.425, "word": "is", "pitch": 167.25, "energy": 0.031}
]
```

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
# Install prosody-specific packages
pip install praat-parselmouth phonemizer pandas

# Install espeak (required for phonemizer)
# macOS:
brew install espeak

# Linux:
sudo apt-get install espeak
```

### 2. Extract Prosody Features
```bash
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips \
  --output-dir data/prosody \
  --n-stress 3
```

**Generated Files:**
- `data/prosody/phoneme_features.json` - Phoneme-level features
- `data/prosody/word_features.csv` - Word-level aggregated features  
- `data/prosody/stress_labels.json` - Stress annotations
- `data/prosody/stress_visualization.png` - Clustering plot

### 3. Train Stress Detection Model (Optional)
```bash
python train_stress_model.py \
  --csv-file data/prosody/word_features.csv \
  --num-epochs 50
```

**Generated:**
- `models/stress_detector.pt` - Trained attention model
- `models/stress_mapping.json` - Label mappings

## 🎨 Demo Mode

Run interactive demos:
```bash
python demo_prosody.py
```

This will:
1. Process a single audio file
2. Show stress detection clustering
3. Display feature statistics
4. Generate visualizations

## 📋 Features Extracted

### Phoneme-Level
- **Phoneme:** IPA symbol or character
- **Timing:** Start, end, duration
- **Pitch:** F0 in Hz (75-600 range)
- **Energy:** RMS energy (0-1)
- **Word:** Parent word context

### Word-Level (Aggregated)
- **Pitch:** avg, max, range
- **Duration:** avg, total
- **Energy:** avg, max
- **Metadata:** num_phonemes

## 🔬 Stress Detection Methods

### Method 1: GMM Clustering (Unsupervised)
```python
from prosody_pipeline import StressDetector
import pandas as pd

df = pd.read_csv('data/prosody/word_features.csv')
detector = StressDetector(n_stress_levels=3)
df = detector.fit(df)
detector.visualize_stress(df, 'plot.png')

print(df[['word', 'avg_pitch', 'avg_duration', 'stress']])
```

**Output:**
```
     word  avg_pitch  avg_duration      stress
0  Hello      185.3         0.180     primary
1    the      120.5         0.065  unstressed
2  world      195.8         0.220     primary
```

### Method 2: Attention Model (Supervised)
```python
from train_stress_model import train_stress_model

train_stress_model(
    csv_file='data/prosody/word_features.csv',
    batch_size=16,
    num_epochs=50
)
```

Uses transformer attention to learn stress from prosodic patterns.

## 🎯 Use Cases

1. **Prosody-Aware TTS** - Control stress in synthesized speech
2. **Pronunciation Assessment** - Detect incorrect stress in L2 learners
3. **Emotion Recognition** - Stress patterns correlate with emotion
4. **Linguistic Research** - Analyze stress across languages/dialects

## 📁 File Structure

```
DL/
├── prosody_pipeline.py          # Main feature extraction
├── train_stress_model.py        # Attention model training
├── demo_prosody.py              # Interactive demos
├── PROSODY_README.md            # Full documentation
├── data/
│   └── prosody/                 # Generated prosody data
│       ├── phoneme_features.json
│       ├── word_features.csv
│       ├── stress_labels.json
│       └── stress_visualization.png
└── models/
    ├── stress_detector.pt       # Trained model
    └── stress_mapping.json      # Label mappings
```

## 🔧 Command Reference

### Extract Features (Full Dataset)
```bash
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips \
  --output-dir data/prosody \
  --n-stress 3
```

### Extract Features (Test Set)
```bash
python prosody_pipeline.py \
  --data-json data/test.json \
  --audio-dir data_source/en/clips \
  --output-dir data/prosody_test \
  --n-stress 3
```

### Train Model
```bash
python train_stress_model.py \
  --csv-file data/prosody/word_features.csv \
  --batch-size 16 \
  --num-epochs 50
```

### Run Demos
```bash
python demo_prosody.py
```

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| GMM Accuracy | 70-85% |
| Attention Model Accuracy | 85-95% |
| Processing Speed | 1-2 sec/utterance |
| Features per 1min audio | ~500-800 phonemes |

## 🐛 Troubleshooting

**Error: "espeak not found"**
```bash
# Install espeak
brew install espeak  # macOS
sudo apt-get install espeak  # Linux
```

**Error: "parselmouth import failed"**
```bash
pip install praat-parselmouth
```

**Low clustering accuracy:**
- Use 3-level stress (primary/secondary/unstressed)
- Check audio quality (16kHz, clear speech)
- Increase dataset size (>500 samples recommended)

## 🔄 Integration with Main STT Pipeline

Add prosody after training main model:

```bash
# 1. Train main STT model
python src/train.py --config configs/config.json

# 2. Extract prosody features
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips

# 3. Train stress detector
python train_stress_model.py

# 4. Use both for prosody-aware applications
```

## 📚 Learn More

- **Full Documentation:** [PROSODY_README.md](PROSODY_README.md)
- **Main Project:** [README.md](README.md)
- **Praat/Parselmouth:** https://parselmouth.readthedocs.io/
- **Phonemizer:** https://github.com/bootphon/phonemizer

## ✅ Validation Checklist

Before using outputs:

- [ ] Espeak installed and working
- [ ] Audio files at 16kHz sample rate
- [ ] Phoneme features contain valid pitch values (>0)
- [ ] Word features CSV has no NaN values
- [ ] Stress labels distributed across all classes
- [ ] Visualization shows clear clusters

## 🎓 Next Steps

1. **Run demos** to understand outputs
2. **Process full dataset** (~20min for 1000 files)
3. **Train attention model** for better accuracy
4. **Integrate with TTS** for prosody control
5. **Expand to multi-language** stress patterns

---

**Created:** January 13, 2026  
**Part of:** Speech-to-Text Transformer Project
