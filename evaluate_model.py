"""
Model Evaluation Script
Evaluates the trained Speech-to-Text model on test dataset
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

from src.model import SpeechToTextTransformer
from src.dataset import TextTokenizer, AudioPreprocessor
from src.utils import calculate_wer, calculate_cer, save_predictions


class ModelEvaluator:
    """Evaluate Speech-to-Text model on test set"""

    def __init__(self, model_path, config_path, vocab_path, device=None):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            vocab_path: Path to vocabulary file
            device: Device to run evaluation on (None for auto-detect)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Device setup
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = TextTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        print(f"Loaded vocabulary with {len(self.tokenizer)} tokens")

        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mels=self.config.get('input_dim', 80)
        )

        # Load model
        self.model = SpeechToTextTransformer(
            input_dim=self.config['input_dim'],
            d_model=self.config['d_model'],
            num_encoder_layers=self.config['num_encoder_layers'],
            num_decoder_layers=self.config['num_decoder_layers'],
            num_heads=self.config['num_heads'],
            d_ff=self.config['d_ff'],
            vocab_size=len(self.tokenizer),
            max_seq_length=self.config['max_seq_length'],
            dropout=0.0  # No dropout during evaluation
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs\n")

    def transcribe_file(self, audio_path, max_len=100):
        """
        Transcribe a single audio file

        Args:
            audio_path: Path to audio file
            max_len: Maximum generation length

        Returns:
            Transcribed text
        """
        # Load and preprocess audio
        waveform, _ = self.preprocessor.load_audio(audio_path)
        audio_features = self.preprocessor.extract_features(waveform)
        audio_features = self.preprocessor.normalize(audio_features)

        # Add batch dimension
        audio_features = audio_features.unsqueeze(0).to(self.device)

        # Generate transcription
        with torch.no_grad():
            predictions = self.model.greedy_decode(
                audio_features,
                max_len=max_len,
                start_token=self.tokenizer.char2idx['<SOS>'],
                end_token=self.tokenizer.char2idx['<EOS>']
            )

        # Decode to text
        transcription = self.tokenizer.decode(predictions[0])

        return transcription

    def evaluate(self, test_file, audio_dir, output_file=None):
        """
        Evaluate model on test dataset

        Args:
            test_file: Path to test data JSON file
            audio_dir: Directory containing audio files
            output_file: Optional path to save detailed results

        Returns:
            Dictionary with evaluation metrics
        """
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        print(f"Evaluating on {len(test_data)} test samples...")
        print("=" * 70)

        predictions = []
        references = []
        audio_dir = Path(audio_dir)

        # Transcribe all test samples
        for sample in tqdm(test_data, desc="Evaluating"):
            audio_path = audio_dir / sample['audio']

            if not audio_path.exists():
                print(f"\nWarning: Audio file not found: {audio_path}")
                continue

            try:
                # Get prediction
                pred_text = self.transcribe_file(str(audio_path))
                ref_text = sample['text']

                predictions.append(pred_text)
                references.append(ref_text)

            except Exception as e:
                print(f"\nError processing {sample['audio']}: {e}")
                continue

        # Calculate metrics
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)

        wer = calculate_wer(references, predictions)
        cer = calculate_cer(references, predictions)

        print(f"\nWord Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
        print(f"Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")

        # Show some examples
        print("\n" + "=" * 70)
        print("Sample Predictions (first 5)")
        print("=" * 70)

        for i in range(min(5, len(predictions))):
            print(f"\n{i+1}. Reference: {references[i]}")
            print(f"   Prediction: {predictions[i]}")
            sample_wer = calculate_wer([references[i]], [predictions[i]])
            print(f"   WER: {sample_wer:.4f}")

        # Save detailed results if requested
        if output_file:
            save_predictions(predictions, references, output_file)
            print(f"\nDetailed results saved to: {output_file}")

        results = {
            'num_samples': len(predictions),
            'wer': wer,
            'cer': cer,
            'predictions': predictions,
            'references': references
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Speech-to-Text Model')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )

    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary file'
    )

    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to test data JSON file'
    )

    parser.add_argument(
        '--audio-dir',
        type=str,
        help='Directory containing audio files (default: from config)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Path to save detailed results (default: evaluation_results.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    args = parser.parse_args()

    # Get audio directory
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)
        audio_dir = config.get('audio_dir', 'data/audio')

    # Initialize evaluator
    evaluator = ModelEvaluator(
        args.model,
        args.config,
        args.vocab,
        args.device
    )

    # Run evaluation
    results = evaluator.evaluate(
        args.test_file,
        audio_dir,
        args.output
    )

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
