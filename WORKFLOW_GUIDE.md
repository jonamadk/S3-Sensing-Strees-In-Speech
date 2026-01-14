# Complete Workflow Guide

This guide walks you through the complete process of creating a Speech-to-Text model using Whisper for automatic transcription.

## Overview

The workflow consists of 4 main steps:

1. **Data Preparation** - Transcribe audio files using Whisper
2. **Training** - Train the transformer model
3. **Evaluation** - Test the model performance
4. **Inference** - Use the model for predictions

## Prerequisites

### 1. Install Dependencies

```bash
# Activate virtual environment
source sst/bin/activate

# Install all dependencies including Whisper
pip install -r requirements.txt

# Install ffmpeg (required for audio processing)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

### 2. Prepare Your Audio Files

Organize your audio files in a directory:

```
my_audio_dataset/
├── recording1.wav
├── recording2.mp3
├── recording3.flac
├── ...
```

Supported formats: WAV, MP3, FLAC, M4A, OGG, OPUS, WMA

## Option 1: Automated Pipeline (Recommended)

### One-Command Solution

Run everything with a single command:

```bash
python pipeline.py --audio-dir path/to/my_audio_dataset
```

This will:
- ✅ Transcribe all audio files using Whisper (base model)
- ✅ Split data into train (80%), validation (10%), test (10%)
- ✅ Create vocab.json from transcriptions
- ✅ Train the transformer model
- ✅ Evaluate on test set
- ✅ Save best model checkpoint

### Custom Pipeline Options

```bash
# Use better Whisper model for more accurate transcriptions
python pipeline.py \
  --audio-dir path/to/audio \
  --whisper-model medium

# Custom data splits
python pipeline.py \
  --audio-dir path/to/audio \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1

# Only prepare dataset (useful for checking data quality first)
python pipeline.py \
  --audio-dir path/to/audio \
  --skip-training

# Resume training from checkpoint
python pipeline.py \
  --audio-dir path/to/audio \
  --resume models/checkpoints/latest_checkpoint.pt
```

## Option 2: Step-by-Step Process

### Step 1: Prepare Dataset

```bash
# Transcribe audio files using Whisper
python prepare_dataset.py \
  --audio-dir path/to/my_audio_dataset \
  --output-dir data \
  --model-size base \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

**Whisper Model Comparison:**

| Model  | Size  | Speed | Accuracy | Use Case |
|--------|-------|-------|----------|----------|
| tiny   | ~1GB  | Fast  | Basic    | Quick testing |
| base   | ~1GB  | Fast  | Good     | **Recommended default** |
| small  | ~2GB  | Medium| Better   | Higher quality needed |
| medium | ~5GB  | Slow  | High     | Professional use |
| large  | ~10GB | Slower| Best     | Maximum accuracy |

**Output:**
- `data/train.json` - Training data
- `data/val.json` - Validation data
- `data/test.json` - Test data
- `data/failed_transcriptions.json` - Log of any failures

### Step 2: Review Generated Data

Check the quality of transcriptions:

```bash
# View first few training samples
head -20 data/train.json

# Check statistics
python -c "
import json
with open('data/train.json') as f:
    data = json.load(f)
    print(f'Training samples: {len(data)}')
    for i, item in enumerate(data[:3]):
        print(f'{i+1}. {item[\"audio\"]}: {item[\"text\"]}')"
```

### Step 3: Configure Training

Edit `configs/config.json` if needed:

```json
{
  "training": {
    "num_epochs": 100,       // Adjust based on data size
    "batch_size": 32,        // Reduce if GPU memory limited
    "learning_rate": 0.0001  // Lower for fine-tuning
  }
}
```

### Step 4: Train the Model

```bash
python src/train.py --config configs/config.json
```

**Monitor training:**
- Training loss should decrease steadily
- Validation loss should decrease (watch for overfitting)
- WER (Word Error Rate) should improve over epochs

**Example output:**
```
Epoch 10/100
Train Loss: 1.2345
Val Loss: 1.3456
WER: 0.3254
Time: 125.34s
```

### Step 5: Evaluate on Test Set

```bash
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json \
  --output evaluation_results.json
```

**Interpreting Results:**

- **WER < 0.1 (10%)** - Excellent performance
- **WER 0.1-0.3** - Good performance
- **WER 0.3-0.5** - Acceptable, may need improvement
- **WER > 0.5** - Poor, needs more training or better data

### Step 6: Use for Inference

```bash
# Transcribe a single file
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio new_audio.wav

# Real-time transcription from microphone
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --realtime
```

## Tips for Better Results

### Data Quality

1. **Audio Quality**
   - Use clear, noise-free recordings
   - Consistent volume levels
   - Sample rate: 16kHz recommended

2. **Dataset Size**
   - Minimum: 100 samples
   - Recommended: 1,000+ samples
   - Professional: 10,000+ samples

3. **Transcription Accuracy**
   - Review Whisper transcriptions for accuracy
   - Fix any obvious errors in JSON files
   - Remove poor quality audio files

### Training Optimization

1. **Start Small**
   ```bash
   # Use smaller model for testing
   python pipeline.py \
     --audio-dir path/to/small_subset \
     --whisper-model tiny
   ```

2. **GPU Acceleration**
   - Training on GPU is 10-100x faster
   - Use `nvidia-smi` to monitor GPU usage
   - Reduce batch size if running out of memory

3. **Hyperparameter Tuning**
   - Try different learning rates: [1e-5, 1e-4, 1e-3]
   - Adjust model size in config.json
   - Experiment with dropout rates

### Common Issues

**Issue: Whisper transcription slow**
- Solution: Use smaller model (tiny/base) or enable GPU

**Issue: Training loss not decreasing**
- Solution: Lower learning rate, check data quality

**Issue: High WER on test set**
- Solution: More training data, longer training, better Whisper model

**Issue: GPU out of memory**
- Solution: Reduce batch_size in config.json

**Issue: Poor transcription quality**
- Solution: Use beam search decoding:
  ```bash
  python src/inference.py ... --beam-width 5
  ```

## Example Workflow

Here's a complete example with a custom dataset:

```bash
# 1. Activate environment
source sst/bin/activate

# 2. Prepare dataset with good Whisper model
python prepare_dataset.py \
  --audio-dir ~/my_recordings \
  --output-dir data \
  --model-size small \
  --train-ratio 0.75 \
  --val-ratio 0.15 \
  --test-ratio 0.10

# 3. Review first 5 transcriptions
head -30 data/train.json

# 4. Train model
python src/train.py --config configs/config.json

# 5. Evaluate
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json

# 6. Test on new audio
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio ~/Downloads/new_recording.wav
```

## Production Deployment

Once you have a trained model:

1. **Export Model**
   - Keep `best_checkpoint.pt`
   - Keep `vocab.json`
   - Keep `config.json`

2. **Create Inference API**
   ```python
   from src.inference import SpeechToTextInference
   
   # Initialize once
   model = SpeechToTextInference(
       'models/checkpoints/best_checkpoint.pt',
       'configs/config.json',
       'data/vocab.json'
   )
   
   # Use for multiple predictions
   text = model.transcribe_file('audio.wav')
   ```

3. **Optimize for Speed**
   - Use smaller model architecture
   - Enable GPU inference
   - Batch multiple requests

## Next Steps

- **Improve accuracy**: Collect more diverse training data
- **Multi-language**: Train separate models per language
- **Fine-tuning**: Train on domain-specific data
- **Real-time**: Optimize for streaming audio

## Support

For issues:
1. Check this guide first
2. Review error messages carefully
3. Ensure all dependencies installed
4. Verify audio file formats supported

Happy training! 🚀
