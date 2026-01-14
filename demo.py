"""
Demo Script - Quick demonstration of the Speech-to-Text system
Shows how to use the complete pipeline with sample data
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print_section("Speech-to-Text Transformer - Demo Guide")

    print("This demo will guide you through using the Speech-to-Text system.\n")

    # Check if we have audio files
    audio_dir = Path("data/audio")

    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav")) + \
            list(audio_dir.glob("*.mp3"))
        if audio_files:
            print(f"✓ Found {len(audio_files)} audio files in data/audio/")
        else:
            print("⚠️  No audio files found in data/audio/")
            print("   Please add some audio files to get started.")
    else:
        print("⚠️  data/audio/ directory not found")
        print("   Creating directory...")
        audio_dir.mkdir(parents=True, exist_ok=True)
        print("   Please add audio files to data/audio/")

    print_section("Option 1: Complete Automated Pipeline")

    print("Run everything with one command:")
    print()
    print("  python pipeline.py --audio-dir data/audio")
    print()
    print("This will:")
    print("  1. Transcribe all audio files using Whisper")
    print("  2. Create train/val/test datasets (80/10/10 split)")
    print("  3. Train the transformer model")
    print("  4. Evaluate on test set")
    print()
    print("Estimated time: 10 minutes - 2 hours (depends on data size)")
    print()

    print_section("Option 2: Step-by-Step Process")

    print("For more control, run each step separately:\n")

    print("Step 1: Prepare Dataset")
    print("  python prepare_dataset.py \\")
    print("    --audio-dir data/audio \\")
    print("    --model-size base")
    print()

    print("Step 2: Train Model")
    print("  python src/train.py --config configs/config.json")
    print()

    print("Step 3: Evaluate Model")
    print("  python evaluate_model.py \\")
    print("    --model models/checkpoints/best_checkpoint.pt \\")
    print("    --config configs/config.json \\")
    print("    --vocab data/vocab.json \\")
    print("    --test-file data/test.json")
    print()

    print("Step 4: Use for Inference")
    print("  python src/inference.py \\")
    print("    --model models/checkpoints/best_checkpoint.pt \\")
    print("    --config configs/config.json \\")
    print("    --vocab data/vocab.json \\")
    print("    --audio your_audio.wav")
    print()

    print_section("Quick Testing (Small Model)")

    print("To test quickly with a small model:")
    print()
    print("1. Use tiny Whisper model:")
    print("   python prepare_dataset.py \\")
    print("     --audio-dir data/audio \\")
    print("     --model-size tiny")
    print()
    print("2. Edit configs/config.json to use smaller model:")
    print('   "d_model": 256,')
    print('   "num_encoder_layers": 4,')
    print('   "num_decoder_layers": 4,')
    print('   "num_epochs": 10,')
    print()
    print("3. Train:")
    print("   python src/train.py --config configs/config.json")
    print()

    print_section("Whisper Model Selection")

    print("Choose based on your needs:\n")
    print("  tiny   - Fastest, for testing (1-2 min for 100 files)")
    print("  base   - Recommended default (5-10 min for 100 files)")
    print("  small  - Better quality (15-20 min for 100 files)")
    print("  medium - High quality (30-45 min for 100 files)")
    print("  large  - Best quality (1-2 hours for 100 files)")
    print()

    print_section("Expected Results")

    print("Training progress:")
    print("  Epoch 1:   Train Loss: ~3.5, Val Loss: ~3.2, WER: ~0.8")
    print("  Epoch 10:  Train Loss: ~1.5, Val Loss: ~1.8, WER: ~0.4")
    print("  Epoch 50:  Train Loss: ~0.5, Val Loss: ~0.8, WER: ~0.2")
    print("  Epoch 100: Train Loss: ~0.2, Val Loss: ~0.5, WER: ~0.1")
    print()
    print("Good final WER: < 0.3 (30% error rate)")
    print("Excellent WER:  < 0.1 (10% error rate)")
    print()

    print_section("Troubleshooting")

    print("Common issues and solutions:\n")

    print("1. 'ffmpeg not found'")
    print("   → Install: brew install ffmpeg (macOS)")
    print()

    print("2. 'CUDA out of memory'")
    print("   → Reduce batch_size in configs/config.json")
    print("   → Use smaller model (d_model: 256 instead of 512)")
    print()

    print("3. Slow transcription")
    print("   → Use smaller Whisper model (tiny or base)")
    print("   → Enable GPU if available")
    print()

    print("4. High WER (poor accuracy)")
    print("   → Use more training data")
    print("   → Train for more epochs")
    print("   → Use larger Whisper model for better transcriptions")
    print()

    print_section("Sample Workflow for 100 Audio Files")

    print("Timing estimates (on M1 Mac / RTX 3080):\n")
    print("1. Whisper transcription (base):  ~10 minutes")
    print("2. Training (100 epochs):         ~2 hours")
    print("3. Evaluation:                    ~2 minutes")
    print()
    print("Total: ~2.5 hours")
    print()

    print_section("Next Steps")

    print("1. Add your audio files to data/audio/")
    print()
    print("2. Run the pipeline:")
    print("   python pipeline.py --audio-dir data/audio")
    print()
    print("3. Wait for training to complete")
    print()
    print("4. Test your model:")
    print("   python src/inference.py \\")
    print("     --model models/checkpoints/best_checkpoint.pt \\")
    print("     --config configs/config.json \\")
    print("     --vocab data/vocab.json \\")
    print("     --audio test_audio.wav")
    print()

    print_section("Additional Resources")

    print("📖 Detailed Guide:    WORKFLOW_GUIDE.md")
    print("⚡ Quick Reference:   QUICK_REFERENCE.md")
    print("📋 Main README:       README.md")
    print()
    print("For more help, check these documentation files!")
    print()

    print("=" * 80)
    print("Ready to start? Run: python pipeline.py --audio-dir data/audio")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
