# 🎙️ Getting Started with Automatic Transcription

This guide will help you use Whisper to automatically transcribe your audio files and train a custom Speech-to-Text model.

## Prerequisites

1.  Python 3.8+ installed
2.  Virtual environment activated: `source sst/bin/activate`
3.  Dependencies installed: `pip install -r requirements.txt`
4.  FFmpeg installed: `brew install ffmpeg` (macOS)

Verify installation:
```bash
./check_installation.sh
```

## Workflow Options

### Option A: One-Click Pipeline (Recommended for Beginners)

```bash
# 1. Add your audio files to data/audio/
mkdir -p data/audio
# Copy your .wav, .mp3, or other audio files here

# 2. Run complete pipeline
python pipeline.py --audio-dir data/audio
```

**That's it!** The pipeline will:
- Transcribe all audio using Whisper
- Split into train/val/test (80/10/10)
- Build vocabulary
- Train the model
- Evaluate performance

### Option B: Step-by-Step (For More Control)

#### Step 1: Prepare Dataset
```bash
python prepare_dataset.py \
  --audio-dir data/audio \
  --model-size base \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

**Output:**
- `data/train.json` - Training samples
- `data/val.json` - Validation samples
- `data/test.json` - Test samples

#### Step 2: Review Transcriptions
```bash
# Check first 5 training samples
head -30 data/train.json

# View statistics
python -c "
import json
with open('data/train.json') as f:
    data = json.load(f)
    print(f'Total: {len(data)} samples')
    for i in range(min(3, len(data))):
        print(f\"{i+1}. {data[i]['audio']}: {data[i]['text']}\")
"
```

#### Step 3: Train Model
```bash
python src/train.py --config configs/config.json
```

Watch for:
- Decreasing training loss ✓
- Decreasing validation loss ✓
- Improving WER ✓

#### Step 4: Evaluate
```bash
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json
```

#### Step 5: Use for Inference
```bash
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --audio new_recording.wav
```


## Tips for Better Results

### 1. Data Quality
-  Use clear audio (no background noise)
-  Consistent volume levels
-  Good microphone quality
-  16kHz sample rate recommended

### 2. Dataset Size
- Minimum: 100 samples
- Recommended: 500-1000 samples
- Professional: 5000+ samples

### 3. Training Time
Reduce training time:
```bash
# Edit configs/config.json
{
  "num_epochs": 50,  # Reduce from 100
  "batch_size": 64,  # Increase if GPU allows
}
```

### 4. Quick Testing
Test with small dataset first:
```bash
# Create small subset
mkdir -p data/test_audio
# Copy 10-20 files to data/test_audio/

# Quick test
python pipeline.py \
  --audio-dir data/test_audio \
  --whisper-model tiny
```

## Expected Timeline

For **100 audio files** (10 seconds each):

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Whisper (base) | 5 min | 20 min |
| Training (50 epochs) | 30 min | 3 hours |
| Evaluation | 1 min | 3 min |
| **Total** | **~40 min** | **~3.5 hours** |

For **1000 audio files**:
- Whisper: 50 min (GPU) / 3 hours (CPU)
- Training: 5 hours (GPU) / 30 hours (CPU)

## Understanding Output

### During Transcription
```
Transcribing: 100%|████████| 100/100 [05:23<00:00]

Transcription complete!
Successful: 98
Failed: 2
```

### During Training
```
Epoch 10/100
Train Loss: 1.234
Val Loss: 1.456
WER: 0.325 (32.5% error rate)
Time: 125s
```

**Good progress:**
- Loss decreasing ✓
- WER improving (getting lower) ✓
- Val loss close to train loss ✓

### After Evaluation
```
Word Error Rate (WER): 0.1234 (12.34%)
Character Error Rate (CER): 0.0567 (5.67%)

Sample Predictions:
1. Reference:  hello world
   Prediction: hello world
   WER: 0.0000
```

**Interpretation:**
- WER < 0.10 (10%) = Excellent ⭐⭐⭐⭐⭐
- WER 0.10-0.30 = Good ⭐⭐⭐⭐
- WER 0.30-0.50 = Acceptable ⭐⭐⭐
- WER > 0.50 = Needs work ⭐

## Common Issues

### Issue: "ffmpeg not found"
```bash
# Solution: Install ffmpeg
brew install ffmpeg  # macOS
```

### Issue: "CUDA out of memory"
```bash
# Solution: Reduce batch size
# Edit configs/config.json
"batch_size": 16  # Instead of 32
```

### Issue: Slow transcription
```bash
# Solution: Use smaller Whisper model
python pipeline.py \
  --audio-dir data/audio \
  --whisper-model tiny
```

### Issue: High WER (poor accuracy)
**Solutions:**
1. More training data
2. Train longer (more epochs)
3. Better Whisper model (small/medium)
4. Clean audio data
5. Use beam search for inference:
   ```bash
   python src/inference.py ... --beam-width 5
   ```

## Example Workflows

### Scenario 1: Podcast Transcription
```bash
# 1. Extract audio from videos
# 2. Place in data/audio/
# 3. Run with good quality model
python pipeline.py \
  --audio-dir data/audio \
  --whisper-model medium \
  --train-ratio 0.85 \
  --val-ratio 0.10 \
  --test-ratio 0.05
```

### Scenario 2: Voice Commands
```bash
# 1. Collect short voice commands
# 2. Quick training
python pipeline.py \
  --audio-dir voice_commands \
  --whisper-model base

# 3. Real-time recognition
python src/inference.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --realtime
```

### Scenario 3: Research Dataset
```bash
# 1. Large dataset preparation
python prepare_dataset.py \
  --audio-dir large_dataset \
  --model-size large \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1

# 2. Optimize config for large model
# Edit configs/config.json

# 3. Train
python src/train.py --config configs/config.json

# 4. Thorough evaluation
python evaluate_model.py \
  --model models/checkpoints/best_checkpoint.pt \
  --config configs/config.json \
  --vocab data/vocab.json \
  --test-file data/test.json \
  --output detailed_results.json
```

## Next Steps

After training:

1. **Test on new audio:**
   ```bash
   python src/inference.py \
     --model models/checkpoints/best_checkpoint.pt \
     --config configs/config.json \
     --vocab data/vocab.json \
     --audio new_audio.wav
   ```

2. **Improve model:**
   - Collect more diverse data
   - Train for more epochs
   - Use larger model architecture

3. **Deploy:**
   - Create REST API
   - Integrate into application
   - Optimize for production

## Getting Help

1. **Demo guide**: `python demo.py`
2. **Complete guide**: `WORKFLOW_GUIDE.md`
3. **Quick reference**: `QUICK_REFERENCE.md`
4. **Check installation**: `./check_installation.sh`

## Summary

**Minimum viable workflow:**
```bash
# 1. Add audio files to data/audio/
# 2. Run this one command:
python pipeline.py --audio-dir data/audio
# 3. Wait for completion
# 4. Your model is ready!
```

That's it! 🎉

For more control, see the step-by-step options above.

---

**Questions?** Check the documentation files or run `python demo.py` for interactive guidance.
