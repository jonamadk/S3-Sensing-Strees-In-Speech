"""
Complete Pipeline: Dataset Preparation, Training, and Testing
Automates the entire workflow from audio files to trained model
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import time


class Pipeline:
    """Complete training pipeline orchestrator"""

    def __init__(self, audio_dir, config_file='configs/config.json'):
        """
        Initialize pipeline

        Args:
            audio_dir: Directory containing audio files
            config_file: Path to configuration file
        """
        self.audio_dir = Path(audio_dir)
        self.config_file = Path(config_file)
        self.project_root = Path(__file__).parent

        # Load configuration
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        print("=" * 80)
        print("Speech-to-Text Training Pipeline")
        print("=" * 80)
        print(f"Audio directory: {self.audio_dir}")
        print(f"Config file: {self.config_file}")
        print()

    def run_command(self, command, description):
        """
        Run a shell command and handle errors

        Args:
            command: Command to run (as list)
            description: Description of the step

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Step: {description}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(command)}\n")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True
            )
            print(f"\n✓ {description} completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {description} failed with error code {e.returncode}")
            return False
        except Exception as e:
            print(f"\n✗ {description} failed: {e}")
            return False

    def step1_prepare_dataset(self, whisper_model='base', train_ratio=0.8,
                              val_ratio=0.1, test_ratio=0.1):
        """Step 1: Prepare dataset using Whisper transcription"""
        print("\n" + "🎤" * 40)
        print("STEP 1: Dataset Preparation")
        print("🎤" * 40)

        command = [
            sys.executable,
            'prepare_dataset.py',
            '--audio-dir', str(self.audio_dir),
            '--output-dir', 'data',
            '--model-size', whisper_model,
            '--language', 'en',
            '--train-ratio', str(train_ratio),
            '--val-ratio', str(val_ratio),
            '--test-ratio', str(test_ratio)
        ]

        return self.run_command(command, "Dataset Preparation with Whisper")

    def step2_update_config(self):
        """Step 2: Update configuration with actual paths"""
        print("\n" + "⚙️" * 40)
        print("STEP 2: Configuration Update")
        print("⚙️" * 40)

        # Update config with correct paths
        self.config['train_file'] = 'data/train.json'
        self.config['val_file'] = 'data/val.json'
        self.config['audio_dir'] = str(self.audio_dir)
        self.config['vocab_file'] = 'data/vocab.json'

        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✓ Configuration updated at: {self.config_file}")
        return True

    def step3_train_model(self, resume=None):
        """Step 3: Train the transformer model"""
        print("\n" + "🚀" * 40)
        print("STEP 3: Model Training")
        print("🚀" * 40)

        command = [
            sys.executable,
            'src/train.py',
            '--config', str(self.config_file)
        ]

        if resume:
            command.extend(['--resume', resume])

        return self.run_command(command, "Model Training")

    def step4_evaluate_model(self):
        """Step 4: Evaluate the trained model on test set"""
        print("\n" + "📊" * 40)
        print("STEP 4: Model Evaluation")
        print("📊" * 40)

        # Check if test.json exists
        test_file = Path('data/test.json')
        if not test_file.exists():
            print("ℹ️  No test set found, skipping evaluation")
            return True

        command = [
            sys.executable,
            'evaluate_model.py',
            '--model', 'models/checkpoints/best_checkpoint.pt',
            '--config', str(self.config_file),
            '--vocab', 'data/vocab.json',
            '--test-file', str(test_file)
        ]

        return self.run_command(command, "Model Evaluation on Test Set")

    def run(self, whisper_model='base', train_ratio=0.8, val_ratio=0.1,
            test_ratio=0.1, skip_training=False, resume=None):
        """
        Run the complete pipeline

        Args:
            whisper_model: Whisper model size
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            skip_training: Skip training step
            resume: Resume from checkpoint
        """
        start_time = time.time()

        # Step 1: Prepare dataset
        if not self.step1_prepare_dataset(whisper_model, train_ratio, val_ratio, test_ratio):
            print("\n❌ Pipeline failed at dataset preparation")
            return False

        # Step 2: Update configuration
        if not self.step2_update_config():
            print("\n❌ Pipeline failed at configuration update")
            return False

        # Step 3: Train model
        if not skip_training:
            if not self.step3_train_model(resume):
                print("\n❌ Pipeline failed at model training")
                return False
        else:
            print("\n⏭️  Skipping training step as requested")

        # Step 4: Evaluate model
        if not skip_training:
            self.step4_evaluate_model()  # Non-critical, don't fail pipeline

        # Pipeline complete
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "🎉" * 40)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉" * 40)
        print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
        print("\nYour Speech-to-Text model is ready!")
        print("\nTo use it for inference:")
        print("  python src/inference.py \\")
        print("    --model models/checkpoints/best_checkpoint.pt \\")
        print("    --config configs/config.json \\")
        print("    --vocab data/vocab.json \\")
        print("    --audio your_audio.wav")
        print()

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete Speech-to-Text Pipeline: Prepare, Train, and Test'
    )

    parser.add_argument(
        '--audio-dir',
        type=str,
        required=True,
        help='Directory containing audio files for transcription'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.json',
        help='Path to configuration file (default: configs/config.json)'
    )

    parser.add_argument(
        '--whisper-model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size for transcription (default: base)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training data ratio (default: 0.8)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation data ratio (default: 0.1)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test data ratio (default: 0.1)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step (only prepare dataset)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
        return

    # Create and run pipeline
    pipeline = Pipeline(
        audio_dir=args.audio_dir,
        config_file=args.config
    )

    success = pipeline.run(
        whisper_model=args.whisper_model,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        skip_training=args.skip_training,
        resume=args.resume
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
