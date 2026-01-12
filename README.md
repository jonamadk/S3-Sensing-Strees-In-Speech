# Speech-to-Text Transformer

A complete implementation of a transformer-based architecture for Automatic Speech Recognition (ASR). This project converts spoken audio into text using state-of-the-art deep learning techniques.

##  What's New

**Automated Dataset Preparation with Whisper!**
- Automatically transcribe audio files using OpenAI's Whisper
- One-command pipeline from raw audio to trained model
- No manual transcription needed!

## Quick Start (2 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python pipeline.py --audio-dir path/to/your/audio/files
```

That's it! The script will transcribe your audio, prepare the dataset, train the model, and evaluate it.

## Features

✨ **Transformer Architecture**
- Multi-head self-attention mechanism
- Encoder-decoder architecture
- Positional encoding for sequence modeling
- Label smoothing for better generalization

 **Audio Processing**
- Mel-spectrogram feature extraction
- Audio normalization and preprocessing
- Support for various audio formats via torchaudio
- Real-time audio processing capability

 **Training Features**
- Mixed precision training (AMP) for faster training
- Gradient clipping for stable training
- Learning rate scheduling
- Early stopping to prevent overfitting
- Automatic checkpointing (best & latest models)
- WER (Word Error Rate) and CER (Character Error Rate) metrics

 **Inference Options**
- File-based transcription
- Real-time microphone transcription
- Greedy decoding
- Beam search decoding for improved accuracy

## Project Structure

```
DL/
├── prepare_dataset.py       #  dataset preparation
├── pipeline.py              #  Complete automation pipeline
├── evaluate_model.py        #  Model evaluation on test set
├── test_setup.py            # Test installation
├── setup.sh                 # Setup script
├── src/
│   ├── model.py             # Transformer model architecture
│   ├── dataset.py           # Data loading and preprocessing
│   ├── train.py             # Training script
│   ├── inference.py         # Inference and real-time transcription
│   └── utils.py             # Helper functions and metrics
├── configs/
│   └── config.json          # Model and training configuration
├── data/
│   ├── train.json           # Training data manifest
│   ├── val.json             # Validation data manifest
│   ├── test.json            # Test data manifest
│   ├── vocab.json           # Vocabulary (auto-generated)
│   └── audio/               # Audio files directory
├── models/
│   └── checkpoints/         # Model checkpoints
├── sst/                     # Virtual environment
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── WORKFLOW_GUIDE.md        #  Detailed workflow guide
└── QUICK_REFERENCE.md       #  Quick command reference
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

### 3. Install FFmpeg (Required for Audio Processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

### 4. Install PyTorch with CUDA (Optional, for GPU acceleration)

```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (already installed)
# No additional action needed
```

### 5. Verify Installation

```bash
python test_setup.py
```

## Usage Guides

- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Complete step-by-step guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command reference sheet

## Data Preparation

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

### Custom Tokenization

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

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Speech-Transformer](https://arxiv.org/abs/1804.08870) - Speech recognition with transformers
- [SpecAugment](https://arxiv.org/abs/1904.08779) - Data augmentation for speech

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
- PyTorch for deep learning
- torchaudio for audio processing
- transformers architecture inspiration from "Attention Is All You Need"

---

**Happy Training! 🚀**
