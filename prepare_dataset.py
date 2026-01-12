"""
Dataset Preparation Script using Whisper for Transcription
Automatically transcribes audio files and creates train/test datasets
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import whisper
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DatasetPreparator:
    """Prepare dataset using Whisper transcription"""

    def __init__(self, audio_dir, output_dir, model_size='base', language='en', device=None):
        """
        Initialize dataset preparator

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save processed data
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Target language for transcription (en for English)
            device: Device to run Whisper on (None for auto-detect)
        """
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = language

        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")
        print(f"Loading Whisper model: {model_size}")

        # Load Whisper model
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Whisper model loaded successfully!\n")

        # Supported audio formats
        self.audio_extensions = {'.wav', '.mp3',
                                 '.flac', '.m4a', '.ogg', '.opus', '.wma'}

    def find_audio_files(self):
        """Find all audio files in the directory"""
        audio_files = []

        print(f"Searching for audio files in: {self.audio_dir}")

        for ext in self.audio_extensions:
            audio_files.extend(self.audio_dir.rglob(f'*{ext}'))

        print(f"Found {len(audio_files)} audio files\n")
        return sorted(audio_files)

    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file using Whisper

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription result
        """
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                fp16=torch.cuda.is_available(),
                verbose=False
            )

            return {
                'audio': audio_path.name,
                'text': result['text'].strip(),
                'language': result.get('language', self.language),
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'audio': audio_path.name,
                'text': '',
                'language': None,
                'success': False,
                'error': str(e)
            }

    def transcribe_dataset(self, audio_files):
        """
        Transcribe all audio files

        Args:
            audio_files: List of audio file paths

        Returns:
            List of transcription results
        """
        results = []
        failed = []

        print("Starting transcription process...")
        print("=" * 70)

        for audio_path in tqdm(audio_files, desc="Transcribing"):
            result = self.transcribe_audio(audio_path)

            if result['success']:
                results.append({
                    'audio': result['audio'],
                    'text': result['text']
                })
            else:
                failed.append({
                    'audio': result['audio'],
                    'error': result['error']
                })
                print(
                    f"\nFailed to transcribe {result['audio']}: {result['error']}")

        print("\n" + "=" * 70)
        print(f"Transcription complete!")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(failed)}")

        if failed:
            # Save failed transcriptions log
            failed_log = self.output_dir / 'failed_transcriptions.json'
            with open(failed_log, 'w') as f:
                json.dump(failed, f, indent=2)
            print(f"Failed transcriptions logged to: {failed_log}")

        return results

    def split_dataset(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets

        Args:
            data: List of data samples
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_ratio,
            random_state=random_state
        )

        # Second split: separate validation from training
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )

        print(f"\nDataset split:")
        print(
            f"  Training samples: {len(train_data)} ({train_ratio*100:.1f}%)")
        print(f"  Validation samples: {len(val_data)} ({val_ratio*100:.1f}%)")
        print(f"  Test samples: {len(test_data)} ({test_ratio*100:.1f}%)")

        return train_data, val_data, test_data

    def save_dataset(self, train_data, val_data, test_data=None):
        """
        Save dataset splits to JSON files

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data (optional)
        """
        # Save training data
        train_file = self.output_dir / 'train.json'
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved training data to: {train_file}")

        # Save validation data
        val_file = self.output_dir / 'val.json'
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        print(f"Saved validation data to: {val_file}")

        # Save test data if provided
        if test_data:
            test_file = self.output_dir / 'test.json'
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print(f"Saved test data to: {test_file}")

    def generate_statistics(self, data):
        """Generate and display dataset statistics"""
        if not data:
            return

        print("\nDataset Statistics:")
        print("=" * 70)

        # Text length statistics
        text_lengths = [len(item['text']) for item in data]
        word_counts = [len(item['text'].split()) for item in data]

        print(f"Total samples: {len(data)}")
        print(f"\nCharacter count:")
        print(f"  Min: {min(text_lengths)}")
        print(f"  Max: {max(text_lengths)}")
        print(f"  Average: {sum(text_lengths)/len(text_lengths):.1f}")

        print(f"\nWord count:")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")
        print(f"  Average: {sum(word_counts)/len(word_counts):.1f}")

        # Sample texts
        print(f"\nSample transcriptions:")
        for i, item in enumerate(data[:3], 1):
            print(f"  {i}. [{item['audio']}]")
            print(
                f"     {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")

    def prepare(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        Main preparation pipeline

        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random seed for reproducibility
        """
        print("=" * 70)
        print("Dataset Preparation Pipeline")
        print("=" * 70)
        print()

        # Step 1: Find audio files
        audio_files = self.find_audio_files()

        if not audio_files:
            print("No audio files found! Please add audio files to the directory.")
            return

        # Step 2: Transcribe all audio files
        transcribed_data = self.transcribe_dataset(audio_files)

        if not transcribed_data:
            print("No successful transcriptions! Please check your audio files.")
            return

        # Step 3: Generate statistics
        self.generate_statistics(transcribed_data)

        # Step 4: Split dataset
        train_data, val_data, test_data = self.split_dataset(
            transcribed_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )

        # Step 5: Save datasets
        self.save_dataset(train_data, val_data, test_data)

        print("\n" + "=" * 70)
        print("Dataset preparation complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the generated JSON files in the data directory")
        print("2. Update configs/config.json if needed")
        print("3. Run training: python src/train.py --config configs/config.json")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare speech dataset using Whisper transcription'
    )

    parser.add_argument(
        '--audio-dir',
        type=str,
        required=True,
        help='Directory containing audio files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for processed data (default: data)'
    )

    parser.add_argument(
        '--model-size',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )

    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Target language for transcription (default: en)'
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
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
        return

    # Initialize preparator
    preparator = DatasetPreparator(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        language=args.language,
        device=args.device
    )

    # Run preparation pipeline
    preparator.prepare(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )


if __name__ == "__main__":
    main()
