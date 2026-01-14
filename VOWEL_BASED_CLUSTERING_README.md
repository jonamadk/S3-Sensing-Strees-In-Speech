# Vowel-Based Word Stress Clustering

## Overview

This document describes the vowel-based prominence clustering approach for detecting word stress in the ARPAbet phoneme dataset. The method uses linguistic features centered on vowel duration and pitch dynamics to identify stressed vs. unstressed words.

## Key Improvements Over GMM Approach

### Previous Approach (GMM)
- Used simple aggregate features: avg_pitch, max_pitch, avg_energy
- Gaussian Mixture Model with 3 clusters
- Treated all phonemes equally (vowels and consonants)

### New Approach (Vowel-Based + KMeans)
- **Vowel-centric analysis**: Separates vowels from consonants
- **Duration features**: Vowel duration, consonant duration, vowel ratio
- **Pitch dynamics**: Slope, standard deviation, mean absolute difference
- **Contextual features**: Pre-pause, post-pause, position normalization
- **Prominence scoring**: Weighted combination of z-scored features
- **KMeans clustering**: More stable than GMM for this task

## Features Extracted

### 1. **Vowel Detection**
Extended vowel set including:
- ARPAbet vowels: AA, AE, AH0, AH1, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
- IPA symbols: o, ᵻ, a, e, ɐ, ɒ, ɜ

### 2. **Duration Features**
- `vowel_duration`: Total duration of vowel phonemes in word
- `consonant_duration`: Total duration of consonant phonemes
- `vowel_ratio`: Proportion of word duration that is vowels
- `word_duration`: Total word duration

### 3. **Pitch Features**
**All phonemes:**
- `pitch_mean`: Average pitch across word
- `pitch_max`: Maximum pitch
- `pitch_min`: Minimum pitch
- `pitch_range`: Max - min pitch
- `pitch_slope`: (final_pitch - initial_pitch) / duration
- `pitch_std`: Standard deviation of pitch
- `pitch_madiff`: Mean absolute difference between consecutive pitch values

**Vowel-only:**
- `vowel_pitch_mean`: Average pitch on vowel phonemes only
- `vowel_pitch_max`: Maximum pitch on vowels

### 4. **Energy Features**
- `energy_mean`: Average RMS energy
- `energy_max`: Maximum energy

### 5. **Contextual Features**
- `pre_pause`: Duration of silence before word
- `post_pause`: Duration of silence after word
- `pos_norm_start`: Normalized position of word start in utterance [0-1]
- `pos_norm_center`: Normalized position of word center

### 6. **Prominence Score**
Weighted combination of z-scored features:
```
prominence_score = 
    0.5 × vowel_duration_z +
    0.5 × pitch_max_z +
    0.3 × pitch_range_z +
    0.2 × pitch_slope_z +
    0.3 × pre_pause_z +
    0.2 × energy_max_z
```

Z-scores are computed per utterance to capture relative prominence.

## Clustering Methods

### 3-Class Clustering (Primary/Secondary/Unstressed)
```bash
python cluster_word_stress.py \
    --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
    --output-dir data/word_stress_clustered \
    --n-clusters 3
```

**Results:**
- **Primary stress** (57 words, 5.4%): Highest prominence, long vowels, high pitch
  - Examples: "outer", "on", "heroes", "some", "review's"
  - Avg prominence: 2.054
  - Avg vowel duration: 0.161s
  
- **Secondary stress** (476 words, 45.3%): Moderate prominence
  - Examples: "Musician", "Films,", "Lee", "Similar", "Artillery"
  - Avg prominence: 0.515
  - Avg vowel duration: 0.291s
  
- **Unstressed** (518 words, 49.3%): Low prominence, short vowels, low pitch
  - Examples: "Secondly,", "When", "In", "The", "of", "and"
  - Avg prominence: -0.699
  - Avg vowel duration: 0.148s

### 2-Class Clustering (Stressed/Unstressed)
```bash
python cluster_word_stress.py \
    --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
    --output-dir data/word_stress_binary \
    --n-clusters 2
```

**Results:**
- **Stressed** (65 words, 6.2%): High prominence
  - Examples: "Originally,", "wording", "year", "throughout", "required"
  - Avg prominence: 2.034
  
- **Unstressed** (986 words, 93.8%): Low prominence
  - Examples: "Secondly,", "When", "In", "The", "Most"
  - Avg prominence: -0.134

## Output Files

### CSV Format
File: `word_stress_features.csv`

42 columns including:
- Word identification: word, phonemes, sentence
- Timing: word_start, word_end, word_duration
- Phoneme counts: num_phonemes, num_vowels
- Raw features: vowel_duration, vowel_ratio, pitch_mean, pitch_max, etc.
- Z-scored features: *_z versions for normalization
- Clustering: prominence_score, cluster, stress_label

### JSON Format
File: `word_stress_features.json`

Same data in JSON array format for easy programmatic access.

### Visualizations
- `word_stress_visualization.png`: 4-panel overview
  - Vowel duration vs pitch max (scatter)
  - Prominence score distribution (box plot)
  - Vowel ratio distribution (box plot)
  - Stress level counts (bar chart)
  
- `word_stress_detailed.png`: Detailed scatter plot
  - X-axis: Max pitch
  - Y-axis: Prominence score
  - Bubble size: Vowel ratio

## Linguistic Insights

### Why This Approach Works

1. **Vowel Duration**: Stressed syllables have longer vowels (English stress timing)
2. **Pitch Peak**: Stressed syllables often have pitch accents (high pitch)
3. **Pausal Context**: Pauses before words signal importance/prominence
4. **Vowel Ratio**: Higher ratio indicates more sonorant content (typical of stress)

### Limitations

- Assumes voiced segments (zero pitch for unvoiced consonants)
- Works best with natural speech prosody
- May misclassify function words in emphasized positions
- Pause detection depends on word segmentation quality

## Technical Details

### Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

### Algorithm Flow
1. Load phoneme-level features (pitch, energy, timing)
2. Group consecutive phonemes by word instance
3. Separate vowels from consonants using extended VOWELS set
4. Compute duration features (vowel/consonant/ratio)
5. Extract pitch dynamics (slope, std, madiff)
6. Detect pauses (pre/post) from word timeline gaps
7. Z-score normalize features per utterance
8. Compute prominence score (weighted combination)
9. KMeans clustering on scaled features
10. Map clusters to stress labels by prominence

### Code Structure
- `VOWELS`: Set of vowel phonemes (ARPAbet + IPA)
- `is_vowel(ph)`: Check if phoneme is vowel
- `safe_pitch(p)`: Filter invalid pitch values
- `aggregate_word_features()`: Extract all features
- `cluster_stress_levels()`: KMeans clustering + labeling
- `visualize_stress_clusters()`: Generate plots

## Future Enhancements

1. **Syllable-level clustering**: Group phonemes by syllables instead of words
2. **Duration normalization**: Account for speaking rate differences
3. **Spectral features**: Add formant frequencies for vowel quality
4. **Temporal features**: Add velocity/acceleration of pitch contours
5. **Semi-supervised learning**: Use IPA stress markers (ˈ, ˌ) as weak labels
6. **Contextual embeddings**: Consider neighboring word stress patterns

## References

- ARPAbet phoneme set: CMU Pronouncing Dictionary format
- Praat F0 algorithm: Parselmouth implementation
- Prominence scoring: Based on Rosenberg (2010) AuToBI features
- Stress detection: Inspired by prosodic analysis in speech synthesis
