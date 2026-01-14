# Prosody Feature Extraction & Stress Detection Pipeline

## 🎯 Overview

This pipeline extracts **prosodic features** (pitch, duration, energy) from audio aligned with phoneme-level transcriptions and builds an **attention-based stress detection model**. The system can identify stressed vs unstressed syllables/words for applications in TTS, pronunciation assessment, and prosody modeling.

## 🌟 Features

### 1. **Prosody Feature Extraction** (`prosody_pipeline.py`)
- **Pitch contour extraction** using Praat (Parselmouth) or librosa
- **Energy/intensity** calculation
- **Duration analysis** at phoneme and word level
- **Phonetic transcription** using espeak phonemizer
- **Feature alignment** with text at character/phoneme level

### 2. **Stress Detection** (Clustering & Attention Model)
- **Gaussian Mixture Model (GMM)** clustering for unsupervised stress detection
- **Attention-based neural model** for supervised stress classification
- **Visualization** of stress patterns in pitch-duration space
- Support for **2-level** (stressed/unstressed) or **3-level** (primary/secondary/unstressed) stress

### 3. **Output Format**
Generates phoneme-level aligned features similar to:
```json
[
  {
    "phoneme": "IH",
    "start": 0.942,
    "end": 1.163,
    "duration": 0.221,
    "word": "It",
    "pitch": 147.1,
    "energy": 0.0234
  },
  {
    "phoneme": "T",
    "start": 1.163,
    "end": 1.184,
    "word": "It",
    "pitch": 158.15,
    "energy": 0.0198
  }
]
```

## 📦 Installation

### Install Additional Dependencies
```bash
# Install prosody-specific packages
pip install praat-parselmouth phonemizer pandas

# Install espeak for phonemizer (required)
# macOS:
brew install espeak

# Linux:
sudo apt-get install espeak espeak-data

# Windows: Download from http://espeak.sourceforge.net/
```

### Verify Installation
```bash
python -c "import parselmouth; import phonemizer; print('✓ Prosody tools installed')"
```

## 🚀 Quick Start

### Step 1: Extract Prosody Features
```bash
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips \
  --output-dir data/prosody \
  --n-stress 3
```

**Outputs:**
- `data/prosody/phoneme_features.json` - Phoneme-level aligned features
- `data/prosody/word_features.csv` - Word-level aggregated features
- `data/prosody/stress_labels.json` - Stress annotations per word
- `data/prosody/stress_visualization.png` - Cluster visualization

### Step 2: Train Attention-Based Stress Model
```bash
python train_stress_model.py \
  --csv-file data/prosody/word_features.csv \
  --batch-size 16 \
  --num-epochs 50
```

**Outputs:**
- `models/stress_detector.pt` - Trained stress detection model
- `models/stress_mapping.json` - Stress label mappings

## 📊 Feature Descriptions

### Phoneme-Level Features
| Feature | Description | Range |
|---------|-------------|-------|
| `phoneme` | IPA phoneme symbol | - |
| `start` | Start time (seconds) | 0+ |
| `end` | End time (seconds) | 0+ |
| `duration` | Phoneme duration (seconds) | 0-1 |
| `word` | Parent word | - |
| `pitch` | Average F0 (Hz) | 75-600 |
| `energy` | RMS energy | 0-1 |

### Word-Level Features
| Feature | Description |
|---------|-------------|
| `avg_pitch` | Mean pitch across word phonemes |
| `max_pitch` | Maximum pitch in word |
| `pitch_range` | Pitch variation (max - min) |
| `avg_duration` | Average phoneme duration |
| `total_duration` | Total word duration |
| `avg_energy` | Average energy |
| `max_energy` | Peak energy |
| `num_phonemes` | Number of phonemes |

## 🔬 Stress Detection Methods

### Method 1: GMM Clustering (Unsupervised)
Uses Gaussian Mixture Models to cluster words based on prosodic features:

```python
from prosody_pipeline import ProsodyDatasetBuilder, StressDetector

builder = ProsodyDatasetBuilder()
df_phonemes, df_words = builder.process_dataset('data/train.json', 'audio/dir')

# Automatic stress detection
detector = StressDetector(n_stress_levels=3)
df_words = detector.fit(df_words)

# Visualize
detector.visualize_stress(df_words, save_path='stress_plot.png')
```

**Clusters based on:**
- Average pitch (higher = more stressed)
- Duration (longer = more stressed)
- Energy (louder = more stressed)

### Method 2: Attention Model (Supervised)
Transformer-based model that learns stress patterns:

```python
from train_stress_model import train_stress_model

train_stress_model(
    csv_file='data/prosody/word_features.csv',
    batch_size=16,
    num_epochs=50
)
```

**Architecture:**
- Input: 8 prosodic features per word
- Transformer encoder with multi-head attention
- Output: Stress class (primary/secondary/unstressed)
- Attention weights reveal which features matter most

## 📈 Example Workflow

### Complete Pipeline
```bash
# 1. Extract prosody features from your dataset
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips

# 2. Examine the generated features
head data/prosody/word_features.csv

# 3. Train stress detection model
python train_stress_model.py

# 4. Use trained model for inference (in Python)
import torch
from train_stress_model import AttentionStressDetector

model = AttentionStressDetector(num_classes=3)
model.load_state_dict(torch.load('models/stress_detector.pt'))
# ... extract features from new audio ...
# predictions = model(features)
```

## 🎨 Visualization Example

The pipeline generates scatter plots showing stress clusters:

```
     Primary Stress (red)
          ▲
    250Hz |     ●
          |   ●   ●
          | ●   Secondary (orange)
    200Hz |      ●  ●
          |    ●  ●
    150Hz |  ● ● Unstressed (blue)
          +─────────────────────►
           0.1s  0.2s  0.3s
              Duration
```

## 🔧 Advanced Usage

### Custom Feature Extraction
```python
from prosody_pipeline import ProsodyExtractor

extractor = ProsodyExtractor(sample_rate=16000)

# Extract pitch contour
times, pitches = extractor.extract_pitch_contour('audio.wav')

# Extract energy
times, energy = extractor.extract_energy('audio.wav')

# Get phonemes from text
phonemes = extractor.get_phonemes_from_text("Hello world")
```

### Forced Alignment Integration
For better alignment, integrate with forced alignment tools:

```python
# Install Montreal Forced Aligner (MFA) or Gentle
# Then replace simple alignment with forced alignment results

from prosody_pipeline import ProsodyExtractor

extractor = ProsodyExtractor()

# Use MFA output instead of simple alignment
mfa_alignments = load_mfa_textgrid('aligned.TextGrid')
# Extract features for each aligned segment
```

## 📝 Use Cases

1. **Text-to-Speech (TTS)**
   - Add natural stress patterns to synthetic speech
   - Control prosody in neural TTS models

2. **Pronunciation Assessment**
   - Detect incorrect stress in language learning
   - Provide feedback on prosody

3. **Speech Emotion Recognition**
   - Stress patterns correlate with emotion
   - Use as features for emotion classification

4. **Linguistic Analysis**
   - Study stress patterns across languages
   - Analyze prosodic variation

## 📊 Performance Expectations

| Metric | Value |
|--------|-------|
| GMM Clustering Accuracy | 70-85% |
| Attention Model Accuracy | 85-95% |
| Processing Speed | ~1-2 sec/utterance |
| Memory Usage | ~2GB for 1000 samples |

## 🔄 Integration with Main Pipeline

Add to existing workflow:

```bash
# After training main STT model
python src/train.py --config configs/config.json

# Extract prosody for TTS enhancement
python prosody_pipeline.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips

# Train stress model
python train_stress_model.py
```

## 🐛 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'parselmouth'`
```bash
pip install praat-parselmouth
```

**Issue:** `phonemizer backend 'espeak' not found`
```bash
# macOS
brew install espeak

# Linux
sudo apt-get install espeak
```

**Issue:** Pitch extraction returns all zeros
- Check audio is not silent
- Verify sample rate matches (16kHz)
- Try adjusting pitch range in extractor

## 📚 References

- Praat documentation: https://www.fon.hum.uva.nl/praat/
- Phonemizer: https://github.com/bootphon/phonemizer
- Stress detection literature: Rosenberg (2010), "AutoBI - A tool for automatic ToBI annotation"

## 🎯 Next Steps

1. **Improve alignment** with Montreal Forced Aligner
2. **Add more features** (spectral tilt, jitter, shimmer)
3. **Multi-language support** for stress patterns
4. **Real-time prosody extraction** for live feedback
5. **Integrate with TTS** to control output prosody

---

**Created for:** Speech-to-Text Transformer Project
**Last Updated:** January 13, 2026
