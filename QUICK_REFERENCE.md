# Quick Reference Guide

## Installation
```bash
source sst/bin/activate
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

## Complete Pipeline (One Command)
```bash
python pipeline.py --audio-dir path/to/audio/files
```

## Step-by-Step Commands

### 1. Prepare Dataset
```bash
python prepare_dataset.py \
  --audio-dir path/to/audio \
  --model-size base
```

### 2. Train Model
```bash
python src/train.py --config configs/config.json
```

### 3. Evaluate
```bash
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json
```

### 4. Inference
```bash
# Single file
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio audio.wav

# Real-time
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --realtime
```

## Whisper Models
- `tiny` - Fastest (testing)
- `base` - Recommended default
- `small` - Better quality
- `medium` - High quality
- `large` - Best quality

## Common Options

### Pipeline Options
```bash
--audio-dir          # Audio files directory
--whisper-model      # Whisper model size
--train-ratio        # Training data ratio (default: 0.8)
--val-ratio          # Validation ratio (default: 0.1)
--test-ratio         # Test ratio (default: 0.1)
--skip-training      # Only prepare dataset
--resume             # Resume from checkpoint
```

### Training Options
```bash
--config             # Config file path
--resume             # Resume from checkpoint
```

### Evaluation Options
```bash
--model              # Model checkpoint
--config             # Config file
--vocab              # Vocabulary file
--test-file          # Test data JSON
--output             # Output results file
```

### Inference Options
```bash
--model              # Model checkpoint
--config             # Config file
--vocab              # Vocabulary file
--audio              # Audio file to transcribe
--realtime           # Real-time microphone input
--beam-width         # Beam search width (default: 1)
```

## File Structure
```
data/
  train.json         # Training data
  val.json           # Validation data
  test.json          # Test data
  vocab.json         # Vocabulary
  audio/             # Audio files

models/checkpoints/
  best_checkpoint.pt    # Best model
  latest_checkpoint.pt  # Latest model

configs/
  config.json        # Configuration
```

## Performance Metrics
- **WER < 0.1** - Excellent
- **WER 0.1-0.3** - Good
- **WER 0.3-0.5** - Acceptable
- **WER > 0.5** - Needs improvement

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow transcription | Use smaller Whisper model |
| GPU OOM | Reduce batch_size |
| High WER | More data, longer training |
| Poor quality | Use beam search (--beam-width 5) |

## Key Files

- `pipeline.py` - Complete automation
- `prepare_dataset.py` - Dataset preparation
- `src/train.py` - Model training
- `evaluate_model.py` - Model evaluation
- `src/inference.py` - Predictions
- `configs/config.json` - Configuration
