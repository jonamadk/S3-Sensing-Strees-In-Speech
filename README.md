# S3-Sensing-Stress-In-Speech

This project is an end to end deep learning work that converts spoken audio into text and extracts detailed phoneme-level prosodic features for linguistic analysis. The phoneme level prosodic features such as pitch, energy, pronunciation duration, vowel ratio, consonant presence, utterance behavior, pitch flow, energy flow, pause etc is labeled as primary, secondary and un-stressed using clustering algorithm K-Means, GMM and Hierarchical with statistical validation. Next, The project studies the traditional ML models and State of Art DL models including Multi-head Attention based model to develop the word stress classifier on the labelled dataset.


**Features**
- Automatic phoneme extraction using ARPAbet with IPA mapping
- Vowel-focused prosodic feature analysis (duration, pitch, energy)
- KMeans and Hierarchical clustering for word stress detection (primary/secondary/unstressed)
- Comprehensive statistical validation with 14+ visualizations
- Zero unknown IPA symbols with extended mapping support
- ML Classifiers ( Random Forest, Decision Tree, Naive Bayes, KNN) and Multi Perceptron Neural Network
- Multi-head attention model for both speech to text and word stress classification

**Automated Dataset Preparation with Whisper!**
- Automatically transcribe audio files using OpenAI's Whisper
- One-command pipeline from raw audio to trained model
- No manual transcription needed!
  
- ** Or you can also use the speech to text attention model developed within the project scope **

## Quick Start

### ASR Pipeline (2 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete ASR pipeline
python pipeline.py --audio-dir path/to/your/audio/files
```

### Prosody Analysis Pipeline (4 Commands)

```bash
# 1. Extract phoneme features with prosody
python prosody_arpabet_simple.py \
  --data-json data/train.json \
  --audio-dir data/audio \
  --output-dir data/prosody_arpabet_full \
  --max-samples 500

# 2. Cluster word stress (3-class: primary/secondary/unstressed)
python cluster_word_stress.py \
  --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --output-dir data/word_stress_clustered \
  --n-clusters 3

# 3. Generate statistical analysis and visualizations
python analyze_stress_clustering.py \
  --clustered-dir data/word_stress_clustered \
  --output-dir analysis_results

# 4. Train ML models for stress prediction
python train_stress_models.py \
  --input-json data/word_stress_clustered/word_stress_features.json \
  --output-dir models \
  --test-size 0.2 \
  --random-state 42
```

**Or use the automated wrapper script:**

```bash
# Generate dataset with clustering in one command
./generate_word_stress_from_audio.sh 500

# Then train models
python train_stress_models.py \
  --input-json data/word_stress_500samples/word_stress_features.json \
  --output-dir models_500samples
```

## Features

### 🎤 Speech Recognition

✨ **Transformer Architecture**
- Multi-head self-attention mechanism
- Encoder-decoder architecture
- Positional encoding for sequence modeling
- Label smoothing for better generalization

🔊 **Audio Processing**
- Mel-spectrogram feature extraction
- Audio normalization and preprocessing
- Support for various audio formats via torchaudio
- Real-time audio processing capability

📈 **Training Features**
- Mixed precision training (AMP) for faster training
- Gradient clipping for stable training
- Learning rate scheduling
- Early stopping to prevent overfitting
- Automatic checkpointing (best & latest models)
- WER (Word Error Rate) and CER (Character Error Rate) metrics

🎯 **Inference Options**
- File-based transcription
- Real-time microphone transcription
- Greedy decoding
- Beam search decoding for improved accuracy

### 🎵 Prosody & Word Stress Analysis

🔤 **Phoneme Extraction**
- ARPAbet phoneme representation with extended IPA mapping
- espeak-based word-to-IPA conversion
- 39 unique ARPAbet phonemes (AA, AE, AH, AO, ...)
- Automatic unknown symbol fixing (ɒ, a, e, ɐ, ɜ → ARPAbet)

📊 **Prosodic Features (42 per word)**
- **Duration**: vowel_duration, consonant_duration, vowel_ratio
- **Pitch**: mean, max, range, slope, variability (Praat F0, 75-600 Hz)
- **Energy**: RMS amplitude (mean, max)
- **Pauses**: pre/post pause detection
- **Position**: normalized utterance position

🎯 **Vowel-Based Clustering**
- KMeans clustering with prominence scoring
- 3-class: primary stress, secondary stress, unstressed
- 2-class: stressed vs unstressed (binary)
- Z-score normalization for feature weighting
- Prominence formula: 0.5×vowel_duration + 0.5×pitch_max + 0.3×pitch_range + ...

🤖 **Machine Learning Models**
- **6 trained models**: KNN, Decision Tree, Random Forest, Naive Bayes, Neural Network, XGBoost
- **Best model**: Neural Network (98.73% accuracy, F1=0.9873)
- **Learning curves**: Overfitting detection and validation
- **Feature importance**: prominence_score, vowel_duration, pitch_max
- **Training history plots**: Loss curves and performance metrics
- **Model persistence**: PKL files for deployment

📈 **Statistical Validation**
- **ANOVA & Kruskal-Wallis tests**: Feature significance (p < 1e-28)
- **Clustering metrics**: Silhouette (0.60), Davies-Bouldin, Calinski-Harabasz
- **14+ visualizations**: Vowel distributions, pitch/energy relationships, 3D scatter, ridge plots
- **Reports**: Comprehensive markdown summaries with statistics

🔧 **Data Quality**
- Extended IPA2ARPABET mapping (30+ symbols)
- Retroactive unknown symbol fixing
- Backup creation before modifications
- Zero unknown symbols guarantee

## Project Structure

```
DL/
├── 🎤 ASR (Speech-to-Text)
│   ├── prepare_dataset.py       # Dataset preparation
│   ├── pipeline.py              # Complete automation pipeline
│   ├── evaluate_model.py        # Model evaluation on test set
│   ├── test_setup.py            # Test installation
│   ├── setup.sh                 # Setup script
│   ├── src/
│   │   ├── model.py             # Transformer model architecture
│   │   ├── dataset.py           # Data loading and preprocessing
│   │   ├── train.py             # Training script
│   │   ├── inference.py         # Inference and real-time transcription
│   │   └── utils.py             # Helper functions and metrics
│   ├── configs/
│   │   └── config.json          # Model and training configuration
│   └── models/
│       └── checkpoints/         # Model checkpoints
│
├── 🎵 Prosody & Word Stress Analysis
│   ├── prosody_arpabet_simple.py        # Phoneme extraction with prosody
│   ├── cluster_word_stress.py           # Vowel-based stress clustering
│   ├── analyze_stress_clustering.py     # Statistical analysis & visualizations
│   ├── train_stress_models.py           # ML model training (6 algorithms)
│   ├── fix_unknown_symbols.py           # Data quality fix tool
│   ├── generate_word_stress_from_audio.sh  # Automated pipeline wrapper
│   └── data/
│       ├── prosody_arpabet_full/
│       │   └── phoneme_features_arpabet.json  # 4,562 phonemes with prosody
│       ├── word_stress_clustered/   # 3-class clustering
│       │   ├── word_stress_features.json      # 1,051 words
│       │   ├── word_stress_features.csv
│       │   ├── word_stress_visualization.png
│       │   └── word_stress_detailed.png
│       ├── word_stress_binary/      # 2-class clustering
│       │   ├── word_stress_features.json
│       │   ├── word_stress_features.csv
│       │   └── *.png
│       └── analysis_results/        # Statistical analysis
│           ├── COMPREHENSIVE_ANALYSIS_REPORT.md
│           ├── VISUALIZATION_GUIDE.md
│           ├── ANALYSIS_SUMMARY.md
│           └── 14 visualization charts
│
├── 🤖 Machine Learning Models
│   └── models/                  # Trained ML models
│       ├── best_model.pkl       # Best performing model (Neural Network)
│       ├── knn_model.pkl
│       ├── decision_tree_model.pkl
│       ├── random_forest_model.pkl
│       ├── naive_bayes_model.pkl
│       ├── neural_network_model.pkl
│       ├── xgboost_model.pkl
│       ├── scaler.pkl           # Feature scaler
│       ├── learning_curves.png
│       ├── neural_network_loss.png
│       ├── xgboost_training.png
│       ├── confusion_matrix_*.png
│       ├── model_comparison_*.png
│       ├── ML_MODELS_REPORT.md  # Performance analysis
│       ├── model_results.csv
│       └── model_info.json
│
├── 📊 Shared Data
│   ├── data/
│   │   ├── train.json           # Training data manifest
│   │   ├── val.json             # Validation data manifest
│   │   ├── test.json            # Test data manifest
│   │   ├── vocab.json           # Vocabulary (auto-generated)
│   │   └── audio/               # Audio files directory
│
├── 📚 Documentation
│   ├── README.md                        # This file
│   ├── WORKFLOW_GUIDE.md                # Detailed workflow guide
│   ├── QUICK_REFERENCE.md               # Quick command reference
│   └── WORD_STRESS_GENERATION_GUIDE.md  # Dataset generation guide
│
├── sst/                         # Virtual environment
└── requirements.txt             # Python dependencies
```

## Installation

### 1. Activate Virtual Environment

```bash
cd /Users/manojadhikari/Documents/project/DL
source sst/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install espeak (Required for Phoneme Extraction)

```bash
# macOS
brew install espeak

# Ubuntu/Debian
sudo apt-get install espeak

# Windows
# Download from http://espeak.sourceforge.net/
```

### 4. Install FFmpeg (Required for Audio Processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

### 5. Install PyTorch with CUDA (Optional, for GPU acceleration)

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (already installed)
# No additional action needed
```

### 6. Verify Installation

```bash
python test_setup.py
```

## Usage Guides

- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Complete step-by-step guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference sheet

## Prosody & Word Stress Analysis

### Overview

The prosody analysis pipeline extracts detailed phoneme-level features and clusters words by stress patterns using vowel-based acoustic features.

**Pipeline Flow:**
```
Audio + Text → Phoneme Extraction → Word Aggregation → Clustering → Statistical Analysis
                (espeak + Praat)     (vowel features)    (KMeans)    (ANOVA + metrics)
```

### Step 1: Extract Phoneme Features

```bash
python prosody_arpabet_simple.py \
  --data-json data/train.json \
  --audio-dir data/audio \
  --output-dir data/prosody_arpabet_full \
  --max-samples 100
```

**What it does:**
1. Converts words to IPA using espeak: `"methodology"` → `mɛθədɒlədʒi`
2. Maps IPA to ARPAbet: `mɛθədɒlədʒi` → `M-EH-TH-AH0-D-AO-L-AH0-JH-IY`
3. Extracts pitch (Praat F0, 75-600 Hz) and energy (RMS) for each phoneme
4. Outputs: `phoneme_features_arpabet.json`

**Output format:**
```json
{
  "char": "EH",
  "start": 0.123,
  "end": 0.234,
  "word": "methodology",
  "pitch": 142.50,
  "energy": 0.0234,
  "sentence": "The methodology is sound"
}
```

**Features:**
- **Pitch extraction**: Praat autocorrelation (10ms sampling)
- **Energy extraction**: RMS amplitude via librosa
- **IPA mapping**: 30+ symbols (ɒ, a, e, ɐ, ɜ, ...)
- **Unknown handling**: Symbols wrapped in `?...?` for debugging

### Step 2: Cluster Word Stress

```bash
# 3-class clustering (primary/secondary/unstressed)
python cluster_word_stress.py \
  --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --output-dir data/word_stress_clustered \
  --n-clusters 3

# 2-class clustering (stressed/unstressed)
python cluster_word_stress.py \
  --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --output-dir data/word_stress_binary \
  --n-clusters 2
```

**What it does:**
1. Groups phonemes by word
2. Separates vowels from consonants
3. Computes 42 prosodic features per word
4. Calculates prominence score (z-normalized features)
5. Clusters using KMeans
6. Labels clusters by prominence (highest=primary, lowest=unstressed)
7. Generates visualizations

**42 Features per word:**
- **Duration** (8): total, vowel, consonant, ratios, counts
- **Pitch** (8): mean, max, min, range, std, slope, variability
- **Energy** (2): mean, max
- **Pauses** (2): pre_pause, post_pause
- **Position** (2): normalized start/center in utterance
- **Plus**: vowel-specific pitch features, prominence score

**Prominence scoring formula:**
```python
prominence = (
    0.5 * vowel_duration_z +
    0.5 * pitch_max_z +
    0.3 * pitch_range_z +
    0.2 * pitch_slope_z +
    0.3 * pre_pause_z +
    0.2 * energy_max_z
)
```

**Outputs:**
- `word_stress_features.json`: Full word features with stress labels
- `word_stress_features.csv`: Tabular format for analysis
- `word_stress_visualization.png`: 4-panel clustering overview
- `word_stress_detailed.png`: Extended visualizations

**Example results (3-class):**
```
Primary stress (56 words):
  - Avg vowel duration: 0.210s
  - Avg pitch: 440 Hz
  - Prominence: 2.01
  - Examples: "Originally", "wording", "throughout"

Secondary stress (431 words):
  - Avg vowel duration: 0.330s (LONGEST)
  - Avg pitch: 207 Hz
  - Prominence: 0.60
  - Examples: "Musician", "Artillery", "methodology"

Unstressed (564 words):
  - Avg vowel duration: 0.221s
  - Avg pitch: 87 Hz
  - Prominence: -0.66
  - Examples: "When", "In", "The", "Most"
```

### Step 3: Statistical Analysis

```bash
python analyze_stress_clustering.py \
  --clustered-dir data/word_stress_clustered \
  --output-dir analysis_results
```

**What it does:**
1. Loads clustered data
2. Runs statistical significance tests
3. Computes clustering quality metrics
4. Generates 7 visualization types (14 total for both datasets)
5. Creates comprehensive markdown reports

**Statistical tests:**
- **ANOVA**: Tests if features differ across stress levels
- **Kruskal-Wallis**: Non-parametric alternative
- **Pairwise t-tests**: Between each stress pair
- **Result**: All features p < 1e-28 (highly significant)

**Clustering metrics:**
- **Silhouette Score**: 0.60 (binary), 0.27 (3-class)
- **Davies-Bouldin Index**: Lower = better separation
- **Calinski-Harabasz Score**: Higher = better defined clusters

**14 Visualizations:**
1. Vowel distribution by stress level (4 subplots)
2. Prosodic features heatmap (correlations)
3. Pitch vs energy scatter plots
4. Statistical significance results
5. Clustering quality metrics
6. Feature distributions by stress
7. Advanced: pairplots, ridge plots, 3D scatter

**Generated reports:**
- `COMPREHENSIVE_ANALYSIS_REPORT.md`: Full statistical analysis
- `VISUALIZATION_GUIDE.md`: Explains each chart
- `ANALYSIS_SUMMARY.md`: Quick reference
- Individual PNG files for each visualization

### Step 4: Train ML Models

```bash
python train_stress_models.py \
  --input-json data/word_stress_clustered/word_stress_features.json \
  --output-dir models \
  --test-size 0.2 \
  --random-state 42
```

**What it does:**
1. Trains 6 ML models: K-Nearest Neighbors, Decision Tree, Random Forest, Naive Bayes, Neural Network (MLP), XGBoost
2. Evaluates each model with accuracy, precision, recall, F1-score
3. Generates learning curves to detect overfitting
4. Creates training history plots for Neural Network and XGBoost
5. Saves best model and all trained models as PKL files
6. Produces comprehensive performance report

**Models trained:**
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Decision Tree**: Rule-based classification
- **Random Forest**: Ensemble of decision trees
- **Naive Bayes**: Probabilistic classifier
- **Neural Network (MLP)**: Multi-layer perceptron with 2 hidden layers (100, 50 neurons)
- **XGBoost**: Gradient boosting classifier

**Output files (in `models/` directory):**
- `best_model.pkl`: Best performing model
- `knn_model.pkl`, `decision_tree_model.pkl`, `random_forest_model.pkl`, `naive_bayes_model.pkl`, `neural_network_model.pkl`, `xgboost_model.pkl`
- `scaler.pkl`: Feature scaler for normalization
- `learning_curves.png`: Training vs validation curves for all models
- `neural_network_loss.png`: Loss curve across epochs
- `xgboost_training.png`: Training metrics evolution
- `confusion_matrix_*.png`: Confusion matrices for each model
- `model_comparison_*.png`: Comparative performance charts
- `ML_MODELS_REPORT.md`: Comprehensive analysis report
- `model_results.csv`: Performance metrics table
- `model_info.json`: Model metadata and parameters



### Fixing Unknown Symbols

If you encounter unknown IPA symbols (e.g., `?ɒ?`, `?a?`):

```bash
# Fix all datasets
python fix_unknown_symbols.py --all

# Fix specific files
python fix_unknown_symbols.py \
  --phoneme-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --word-json data/word_stress_clustered/word_stress_features.json \
  --word-csv data/word_stress_clustered/word_stress_features.csv
```

**What it does:**
- Maps unknown IPA symbols to ARPAbet: `?ɒ?→AO`, `?a?→AE`, `?ɜ?→ER`, etc.
- Creates `.backup` files before modification
- Reports number of fixes
- Verifies zero unknown symbols remaining

**Symbol fixes:**
```python
SYMBOL_FIXES = {
    "?ɒ?": "AO",   # British "lot" vowel
    "?a?": "AE",   # open front vowel
    "?ɜ?": "ER",   # "nurse" vowel
    "?ɐ?": "AH0",  # near-open central
    "?e?": "EY"    # close-mid front
}
```

### Understanding Pitch vs Energy

**Pitch (Hz):**
- **Definition**: Fundamental frequency of vocal fold vibration
- **Extraction**: Praat F0 autocorrelation algorithm
- **Range**: 75-600 Hz (filters noise, octave errors)
- **Sampling**: 10ms intervals (time_step=0.01)
- **Use**: Indicates intonation, stress, emotion

**Energy (RMS):**
- **Definition**: Root Mean Square amplitude
- **Formula**: $E = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$
- **Extraction**: librosa RMS on phoneme segment
- **Use**: Indicates loudness, emphasis

**Key difference**: Pitch = frequency (how high/low), Energy = amplitude (how loud)

## Data Preparation (ASR)

### Automatic Transcription with Whisper (Recommended)

**NEW!** Automatically transcribe your audio files using OpenAI's Whisper model:

```bash
# Complete pipeline: transcribe, prepare, train, and evaluate
python pipeline.py --audio-dir path/to/your/audio/files

# Or just prepare the dataset without training
python prepare_dataset.py \
  --audio-dir path/to/your/audio/files \
  --output-dir data \
  --model-size base
```

Supported Whisper models:
- `tiny` - Fastest, least accurate (~1GB)
- `base` - Good balance (default, ~1GB)
- `small` - Better quality (~2GB)
- `medium` - High quality (~5GB)
- `large` - Best quality (~10GB)

The script will:
-  Automatically transcribe all audio files
-  Split into train/val/test sets (80/10/10 by default)
-  Generate `train.json`, `val.json`, and `test.json`
-  Create vocabulary from transcriptions
-  Display dataset statistics

### Manual Data Format

If you already have transcriptions:

**train.json / val.json format:**
```json
[
  {
    "audio": "audio1.wav",
    "text": "transcription of audio one"
  },
  {
    "audio": "audio2.wav",
    "text": "transcription of audio two"
  }
]
```

### Recommended Datasets

- **LibriSpeech**: Free English speech corpus
- **Common Voice**: Multilingual dataset
- **TIMIT**: Phonetic speech corpus
- **Custom recordings**: Record your own data

## Configuration

Edit [configs/config.json](configs/config.json) to customize:

```json
{
  "model": {
    "d_model": 512,              // Model dimension
    "num_encoder_layers": 6,     // Number of encoder layers
    "num_decoder_layers": 6,     // Number of decoder layers
    "num_heads": 8,              // Attention heads
    "d_ff": 2048                 // Feed-forward dimension
  },
  "training": {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0001
  }
}
```

## Training

### Quick Start (Complete Pipeline)

**NEW!** Use the automated pipeline for the entire workflow:

```bash
# Complete pipeline: transcribe, train, and evaluate
python pipeline.py --audio-dir path/to/your/audio/files

# Custom options
python pipeline.py \
  --audio-dir path/to/audio \
  --whisper-model small \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1

# Only prepare dataset (skip training)
python pipeline.py \
  --audio-dir path/to/audio \
  --skip-training
```

### Manual Training

If you already have prepared data:

```bash
python src/train.py --config configs/config.json
```

### Resume Training from Checkpoint

```bash
python src/train.py \
  --config configs/config.json \
  --resume models/checkpoints/latest_checkpoint.pt
```

### Training Output

The training script will:
- Build vocabulary from training data
- Save checkpoints to `models/checkpoints/`
- Display progress with loss and WER metrics
- Save best model based on validation loss

### Monitor Training

```
Epoch 1/100
Train Loss: 2.3456
Val Loss: 2.1234
WER: 0.4523
Time: 125.34s
LR: 0.000100
```

## Inference

### Transcribe Audio File

```bash
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio path/to/audio.wav
```

### Real-time Transcription from Microphone

```bash
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --realtime
```

### Using Beam Search (Better Quality)

```bash
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio path/to/audio.wav \
  --beam-width 5
```

## Model Architecture

### Transformer Components

1. **Audio Encoder**
   - Converts mel-spectrogram to embeddings
   - Multi-head self-attention layers
   - Positional encoding for temporal information

2. **Text Decoder**
   - Character/word-level embeddings
   - Masked self-attention (causal)
   - Cross-attention with encoder outputs

3. **Output Layer**
   - Linear projection to vocabulary size
   - Softmax for token probabilities

### Key Features

- **Multi-Head Attention**: Captures different aspects of relationships
- **Positional Encoding**: Maintains sequence order information
- **Feed-Forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enables deep networks


## Performance Tuning

### For Limited GPU Memory

```json
{
  "batch_size": 8,
  "d_model": 256,
  "num_encoder_layers": 4,
  "num_decoder_layers": 4
}
```

### For Better Accuracy

```json
{
  "batch_size": 32,
  "d_model": 768,
  "num_encoder_layers": 12,
  "num_decoder_layers": 12,
  "beam_width": 10
}
```

### Training Tips

1. **Start small**: Test with small model first
2. **Use pretrained**: Fine-tune on your domain data
3. **Data augmentation**: Add noise, speed perturbation
4. **Learning rate**: Start with 1e-4, adjust based on loss
5. **Batch size**: Larger batches = more stable gradients

## Evaluation Metrics

### Evaluate on Test Set

**NEW!** Evaluate your trained model on the test set:

```bash
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json \
  --output evaluation_results.json
```

This will display:
- Overall WER (Word Error Rate) and CER (Character Error Rate)
- Sample predictions with individual WER scores
- Detailed results saved to JSON file

### Word Error Rate (WER)
```
WER = (Substitutions + Insertions + Deletions) / Total Words
```
Lower is better (0.0 = perfect)

### Character Error Rate (CER)
```
CER = (Char Substitutions + Insertions + Deletions) / Total Chars
```
More fine-grained than WER

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Reduce model dimensions
- Use gradient accumulation
- Enable mixed precision training

### Poor Transcription Quality
- Increase training data
- Train for more epochs
- Use beam search decoding
- Improve audio quality
- Add data augmentation

### Slow Training
- Enable mixed precision (AMP)
- Increase batch size
- Use multiple GPUs
- Reduce sequence lengths

## Advanced Usage

### Prosody Analysis API

```python
from prosody_arpabet_simple import phonemize_word, ipa_to_arpabet, extract_pitch, extract_energy

# Convert word to phonemes
word = "methodology"
ipa = phonemize_word(word)  # Returns: "mɛθədɒlədʒi"
phonemes = ipa_to_arpabet(ipa)  # Returns: ['M', 'EH', 'TH', 'AH0', 'D', 'AO', 'L', 'AH0', 'JH', 'IY']

# Extract prosodic features
pitch = extract_pitch("audio.wav", start=0.0, end=0.5)  # Returns: 142.5 Hz
energy = extract_energy("audio.wav", start=0.0, end=0.5)  # Returns: 0.0234
```

### Clustering Parameters

```python
# In cluster_word_stress.py
VOWELS = set("""
AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW
AA0 AA1 AA2 AE0 AE1 AE2 ...
o ᵻ a e ɐ ɒ ɜ  # IPA extensions
""".split())

# Prominence weights (adjust for your data)
prominence_score = (
    0.5 * vowel_duration_z +
    0.5 * pitch_max_z +
    0.3 * pitch_range_z +
    0.2 * pitch_slope_z +
    0.3 * pre_pause_z +
    0.2 * energy_max_z
)
```

### Custom Tokenization (ASR)

Modify [src/dataset.py](src/dataset.py) to use:
- Word-level tokenization
- BPE (Byte Pair Encoding)
- SentencePiece tokenizer

### Data Augmentation

Add to preprocessing:
- Speed perturbation
- Noise injection
- SpecAugment
- Time stretching

### Multi-GPU Training

```python
# In train.py
model = nn.DataParallel(model)
```

## API Usage

### ASR API

```python
from src.inference import SpeechToTextInference

# Initialize
inference = SpeechToTextInference(
    model_path='models/checkpoints/best_checkpoint.pt',
    config_path='configs/config.json',
    vocab_path='data/vocab.json'
)

# Transcribe
text = inference.transcribe_file('audio.wav')
print(f"Transcription: {text}")
```

### Prosody Analysis API

```python
import json
import pandas as pd

# Load clustered word data
with open('data/word_stress_clustered/word_stress_features.json') as f:
    words = json.load(f)

# Filter by stress level
primary_stress = [w for w in words if w['stress_label'] == 'primary']
print(f"Primary stress words: {len(primary_stress)}")
print(f"Average pitch: {sum(w['pitch_mean'] for w in primary_stress) / len(primary_stress):.1f} Hz")

# Load as DataFrame for analysis
df = pd.read_csv('data/word_stress_clustered/word_stress_features.csv')
print(df.groupby('stress_label')['vowel_duration'].describe())
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{speech2text_transformer,
  title={Speech-to-Text Transformer},
  author={Manoj Adhikari},
  year={2026},
  url={https://github.com/jonamadk/speech-to-text}
}
```

## License

MIT License - feel free to use for research and commercial purposes.

## References

### Speech Recognition
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Speech-Transformer](https://arxiv.org/abs/1804.08870) - Speech recognition with transformers
- [SpecAugment](https://arxiv.org/abs/1904.08779) - Data augmentation for speech

### Prosody & Phonetics
- [Praat](http://www.fon.hum.uva.nl/praat/) - Phonetic analysis software
- [espeak](http://espeak.sourceforge.net/) - Text-to-phoneme converter
- [ARPAbet](https://en.wikipedia.org/wiki/ARPABET) - Phonetic transcription code
- [IPA](https://www.internationalphoneticassociation.org/) - International Phonetic Alphabet
- Vowel-based stress detection using KMeans clustering
- Prosodic feature extraction for linguistic analysis

### Example Commands

```bash
# Generate word stress dataset with 1000 samples
./generate_word_stress_from_audio.sh 1000

# Cluster word stress
python cluster_word_stress.py \
  --input-json data/prosody_500samples/phoneme_features_arpabet.json \
  --output-dir data/word_stress_500samples \
  --n-clusters 3 \
  --method kmeans

# Train attention models
python train_attention_models.py \
  --input-json data/word_stress_500samples/word_stress_features.json \
  --output-dir models_attention \
  --test-size 0.2 \
  --batch-size 64 \
  --epochs 100 \
  --lr 0.001 \
  --random-state 42

# Validate attention generalization
python validate_attention_generalization.py \
  --input-json data/word_stress_500samples/word_stress_features.json \
  --output-dir models_attention \
  --k-folds 5 \
  --bootstrap-iterations 10 \
  --epochs 50
```
## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and documentation
- Provide error messages and system info

## Acknowledgments

Built with:
- **PyTorch** - Deep learning framework
- **torchaudio** - Audio processing
- **librosa** - Audio analysis and feature extraction
- **parselmouth** - Praat integration for pitch extraction
- **espeak** - IPA phoneme generation
- **scikit-learn** - KMeans clustering and metrics
- **seaborn & matplotlib** - Statistical visualizations
- **pandas & numpy** - Data manipulation
- Transformer architecture from "Attention Is All You Need"

Special thanks to the open-source community for these amazing tools!

---

**Happy Training!**
