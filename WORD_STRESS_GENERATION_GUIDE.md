# Word Stress Feature Generation - Usage Guide

## Overview
This project uses the **original dataset generation pipeline** to extract word stress features from audio files. The workflow consists of two main steps:

1. **Phoneme Extraction** → Creates `phoneme_features_arpabet.json`
2. **Word Stress Clustering** → Creates `word_stress_features.json`

## Quick Start

### Option 1: Use the Combined Script (Recommended)

```bash
# Process 100 samples for testing
./generate_word_stress_from_audio.sh 100

# Process ALL samples from train.json
./generate_word_stress_from_audio.sh
```

### Option 2: Run Steps Manually

#### Step 1: Extract Phoneme Features
```bash
python prosody_arpabet_simple.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips \
  --output-dir data/prosody_from_train \
  --max-samples 100  # Optional: limit samples for testing
```

**Output:** `data/prosody_from_train/phoneme_features_arpabet.json`

#### Step 2: Generate Word Stress Features
```bash
python cluster_word_stress.py \
  --input-json data/prosody_from_train/phoneme_features_arpabet.json \
  --output-dir data/word_stress_from_train \
  --n-clusters 3 \
  --method kmeans
```

**Outputs:**
- `data/word_stress_from_train/word_stress_features.json` ← **Main output**
- `data/word_stress_from_train/word_stress_features.csv`
- `data/word_stress_from_train/word_stress_visualization.png`
- `data/word_stress_from_train/word_stress_detailed.png`

## Output Format

### phoneme_features_arpabet.json
```json
[
  {
    "char": "S",
    "start": 0.0,
    "end": 0.074,
    "word": "Secondly,",
    "pitch": 0.0,
    "energy": 0.0005,
    "sentence": "Secondly, the methodology requires..."
  },
  ...
]
```

### word_stress_features.json
```json
[
  {
    "word": "Secondly,",
    "word_start": 0.0,
    "word_end": 0.591,
    "word_duration": 0.591,
    "sentence": "Secondly, the methodology requires...",
    "num_phonemes": 8,
    "num_vowels": 3,
    "phonemes": "SEHKAH0NDLIY",
    "vowel_duration": 0.222,
    "consonant_duration": 0.369,
    "vowel_ratio": 0.3756,
    "pitch_mean": 123.34,
    "pitch_max": 126.16,
    "pitch_min": 120.52,
    "pitch_range": 5.64,
    "pitch_slope": 0.0,
    "pitch_std": 2.82,
    "pitch_madiff": 5.64,
    "vowel_pitch_mean": 120.52,
    "vowel_pitch_max": 120.52,
    "energy_mean": 0.0226,
    "energy_max": 0.0599,
    "pre_pause": 0.0,
    "post_pause": 0.0,
    "pos_norm_start": 0.0,
    "pos_norm_center": 0.0246,
    "prominence_score": -0.4545,
    "stress_label": "unstressed"  ← PRIMARY, SECONDARY, or UNSTRESSED
  },
  ...
]
```

## Processing Time Estimates

- **100 samples**: ~48 seconds (phonemes) + ~2 seconds (clustering) = **~50 seconds**
- **Full dataset (17,199 samples)**: ~2.5 hours (phonemes) + ~30 seconds (clustering) = **~2.5 hours**

## Current Status

✅ **Completed:**
- 100-sample test run → Generated 1,051 words with stress labels
- Outputs in `data/word_stress_from_train/`

🔄 **In Progress:**
- Full dataset extraction running in background
- Monitor: `tail -f prosody_generation.log`
- ETA: ~2.5 hours total

## Clustering Methods

The word stress clustering uses K-means with 3 clusters based on:
- **Vowel duration** (longer = more stressed)
- **Pitch dynamics** (higher pitch = more stressed)
- **Prominence score** (combined weighted features)

Stress levels assigned:
- **Primary** (0): Most prominent syllables
- **Secondary** (1): Moderately prominent
- **Unstressed** (2): Least prominent

## Comparison: Old vs New Scripts

### ✅ USING (Original Pipeline):
1. `prosody_arpabet_simple.py` → Phoneme-level features with ARPAbet
2. `cluster_word_stress.py` → Word-level clustering with stress labels

### ❌ NOT USING:
- `generate_stress_dataset_from_train.py` (new script, simplified approach)

## Next Steps

After full generation completes:
1. Train ML models on `word_stress_features.json`
2. Run correlation analysis
3. Compare with existing `data/word_stress_clustered/word_stress_features.json`

## Example Commands

```bash
# Quick test (100 samples)
./generate_word_stress_from_audio.sh 100

# Full generation
./generate_word_stress_from_audio.sh

# Check progress
tail -f prosody_generation.log

# Train ML models after generation
python train_stress_models.py \
  --input-json data/word_stress_full/word_stress_features.json \
  --output-dir models_from_real_audio \
  --test-size 0.2

# Run correlation analysis
python correlation_analysis.py \
  --input-json data/word_stress_full/word_stress_features.json \
  --output-dir correlation_real_audio
```
