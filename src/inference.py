"""
Inference script for Speech-to-Text Transformer
Handles audio file prediction and real-time transcription
"""

import torch
import argparse
import json
from pathlib import Path
import sounddevice as sd
import numpy as np
from queue import Queue
import sys

from model import SpeechToTextTransformer
from dataset import TextTokenizer, AudioPreprocessor


class SpeechToTextInference:
    """Inference class for Speech-to-Text model"""

    def __init__(self, model_path, config_path, vocab_path, device=None):
        """
        Initialize inference engine

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            vocab_path: Path to vocabulary file
            device: Device to run inference on (None for auto-detect)
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
            dropout=0.0  # No dropout during inference
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")
        print(f"Checkpoint from epoch {checkpoint['epoch']}")

    def transcribe_file(self, audio_path, max_len=100):
        """
        Transcribe an audio file

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

    def transcribe_audio_array(self, audio_array, sample_rate, max_len=100):
        """
        Transcribe audio from numpy array

        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate of audio
            max_len: Maximum generation length

        Returns:
            Transcribed text
        """
        # Convert to torch tensor
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

        # Resample if necessary
        if sample_rate != self.preprocessor.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, self.preprocessor.sample_rate)
            waveform = resampler(waveform)

        # Extract features
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

    def beam_search_decode(self, src, beam_width=5, max_len=100):
        """
        Beam search decoding for better quality transcriptions

        Args:
            src: Audio features [batch_size, seq_len, input_dim]
            beam_width: Beam width for search
            max_len: Maximum generation length

        Returns:
            Best transcription
        """
        self.model.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode audio
        memory = self.model.encode(src)

        # Initialize beam
        start_token = self.tokenizer.char2idx['<SOS>']
        end_token = self.tokenizer.char2idx['<EOS>']

        # Beam: list of (sequence, score)
        beam = [(torch.ones(1, 1).fill_(start_token).long().to(device), 0.0)]

        completed_sequences = []

        for _ in range(max_len):
            all_candidates = []

            for seq, score in beam:
                # If sequence has ended, add to completed
                if seq[0, -1].item() == end_token:
                    completed_sequences.append((seq, score))
                    continue

                # Generate target mask
                tgt_mask = self.model.generate_square_subsequent_mask(
                    seq.size(1)).to(device)

                # Decode
                with torch.no_grad():
                    out = self.model.decode(seq, memory, tgt_mask=tgt_mask)
                    prob = torch.log_softmax(
                        self.model.output_projection(out[:, -1, :]), dim=-1)

                # Get top-k tokens
                topk_probs, topk_indices = torch.topk(prob, beam_width)

                # Add candidates
                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat([seq, next_token], dim=1)
                    next_score = score + topk_probs[0, i].item()
                    all_candidates.append((next_seq, next_score))

            # Select top beam_width candidates
            beam = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
                :beam_width]

            # If all sequences have ended, break
            if len(completed_sequences) >= beam_width:
                break

        # Add remaining sequences to completed
        completed_sequences.extend(beam)

        # Get best sequence
        best_seq, best_score = max(completed_sequences, key=lambda x: x[1])

        return best_seq


class RealTimeTranscriber:
    """Real-time audio transcription from microphone"""

    def __init__(self, inference_engine, chunk_duration=3.0):
        """
        Initialize real-time transcriber

        Args:
            inference_engine: SpeechToTextInference instance
            chunk_duration: Duration of audio chunks in seconds
        """
        self.inference_engine = inference_engine
        self.chunk_duration = chunk_duration
        self.sample_rate = inference_engine.preprocessor.sample_rate
        self.chunk_samples = int(chunk_duration * self.sample_rate)
        self.audio_queue = Queue()

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def start(self):
        """Start real-time transcription"""
        print(f"Starting real-time transcription...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk duration: {self.chunk_duration} seconds")
        print("Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_samples
            ):
                while True:
                    # Get audio chunk
                    audio_chunk = self.audio_queue.get()

                    # Convert to 1D array
                    audio_array = audio_chunk.flatten()

                    # Transcribe
                    try:
                        transcription = self.inference_engine.transcribe_audio_array(
                            audio_array,
                            self.sample_rate
                        )

                        if transcription.strip():
                            print(f"Transcription: {transcription}")
                    except Exception as e:
                        print(f"Error during transcription: {e}")

        except KeyboardInterrupt:
            print("\nStopped real-time transcription")


def main():
    parser = argparse.ArgumentParser(description='Speech-to-Text Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--audio', type=str,
                        help='Path to audio file to transcribe')
    parser.add_argument('--realtime', action='store_true',
                        help='Enable real-time transcription')
    parser.add_argument('--beam-width', type=int, default=1,
                        help='Beam width (1 for greedy)')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Initialize inference engine
    inference = SpeechToTextInference(
        args.model,
        args.config,
        args.vocab,
        args.device
    )

    if args.realtime:
        # Real-time transcription
        transcriber = RealTimeTranscriber(inference)
        transcriber.start()

    elif args.audio:
        # File transcription
        print(f"Transcribing: {args.audio}")
        transcription = inference.transcribe_file(args.audio)
        print(f"\nTranscription: {transcription}")

    else:
        print("Please specify --audio for file transcription or --realtime for live transcription")


if __name__ == "__main__":
    main()
