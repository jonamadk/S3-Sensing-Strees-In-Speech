"""
Dataset and data preprocessing utilities for Speech-to-Text
Handles audio loading, feature extraction, and text tokenization
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import json
import os
from pathlib import Path


class TextTokenizer:
    """Simple character-level tokenizer for text"""

    def __init__(self, vocab_file=None):
        """
        Initialize tokenizer

        Args:
            vocab_file: Path to vocabulary file (JSON)
        """
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<SOS>'
        self.END_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Initialize with special tokens
            self.char2idx = {
                self.PAD_TOKEN: 0,
                self.START_TOKEN: 1,
                self.END_TOKEN: 2,
                self.UNK_TOKEN: 3,
            }
            self.idx2char = {v: k for k, v in self.char2idx.items()}

    def build_vocab(self, texts):
        """
        Build vocabulary from list of texts

        Args:
            texts: List of text strings
        """
        # Get all unique characters
        chars = set()
        for text in texts:
            chars.update(text.lower())

        # Add to vocabulary (keeping special tokens)
        for char in sorted(chars):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs

        Args:
            text: Input text string
            add_special_tokens: Whether to add START and END tokens

        Returns:
            List of token IDs
        """
        text = text.lower()
        tokens = [self.char2idx.get(
            char, self.char2idx[self.UNK_TOKEN]) for char in text]

        if add_special_tokens:
            tokens = [self.char2idx[self.START_TOKEN]] + \
                tokens + [self.char2idx[self.END_TOKEN]]

        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().numpy()

        chars = []
        special_tokens = {self.PAD_TOKEN, self.START_TOKEN,
                          self.END_TOKEN, self.UNK_TOKEN}

        for idx in token_ids:
            char = self.idx2char.get(int(idx), self.UNK_TOKEN)
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)

        return ''.join(chars)

    def save_vocab(self, filepath):
        """Save vocabulary to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {str(k): v for k, v in self.idx2char.items()}
            }, f, indent=2)

    def load_vocab(self, filepath):
        """Load vocabulary from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}

    def __len__(self):
        return len(self.char2idx)


class AudioPreprocessor:
    """Preprocesses audio files to mel-spectrogram features"""

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400,
                 hop_length=160, win_length=400):
        """
        Initialize audio preprocessor

        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            normalized=True
        )

        # Amplitude to dB conversion
        self.amplitude_to_db = T.AmplitudeToDB()

    def load_audio(self, filepath):
        """
        Load audio file and resample if necessary

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sr = torchaudio.load(filepath)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, self.sample_rate

    def extract_features(self, waveform):
        """
        Extract mel-spectrogram features from waveform

        Args:
            waveform: Audio waveform tensor [1, num_samples]

        Returns:
            Mel-spectrogram features [n_mels, time_steps]
        """
        # Compute mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Remove channel dimension and transpose to [time_steps, n_mels]
        mel_spec_db = mel_spec_db.squeeze(0).transpose(0, 1)

        return mel_spec_db

    def normalize(self, features, mean=None, std=None):
        """
        Normalize features

        Args:
            features: Feature tensor
            mean: Mean for normalization (computed if None)
            std: Std for normalization (computed if None)

        Returns:
            Normalized features
        """
        if mean is None:
            mean = features.mean()
        if std is None:
            std = features.std()

        return (features - mean) / (std + 1e-8)


class SpeechToTextDataset(Dataset):
    """Dataset for Speech-to-Text training"""

    def __init__(self, data_file, audio_dir, tokenizer, preprocessor,
                 max_audio_len=None, max_text_len=None):
        """
        Initialize dataset

        Args:
            data_file: Path to JSON file with audio paths and transcriptions
                      Format: [{"audio": "path/to/audio.wav", "text": "transcription"}, ...]
            audio_dir: Base directory for audio files
            tokenizer: TextTokenizer instance
            preprocessor: AudioPreprocessor instance
            max_audio_len: Maximum audio length in frames
            max_text_len: Maximum text length in tokens
        """
        self.audio_dir = Path(audio_dir)
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Load data
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            Dictionary with 'audio_features', 'text_tokens', 'text'
        """
        sample = self.data[idx]

        # Load and process audio
        audio_path = self.audio_dir / sample['audio']
        waveform, _ = self.preprocessor.load_audio(str(audio_path))
        audio_features = self.preprocessor.extract_features(waveform)
        audio_features = self.preprocessor.normalize(audio_features)

        # Truncate if necessary
        if self.max_audio_len and audio_features.size(0) > self.max_audio_len:
            audio_features = audio_features[:self.max_audio_len]

        # Process text
        text = sample['text']
        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate if necessary
        if self.max_text_len and len(text_tokens) > self.max_text_len:
            text_tokens = text_tokens[:self.max_text_len]

        return {
            'audio_features': audio_features,
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'text': text,
            'audio_len': audio_features.size(0),
            'text_len': len(text_tokens)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched and padded tensors
    """
    # Get maximum lengths in batch
    max_audio_len = max(sample['audio_len'] for sample in batch)
    max_text_len = max(sample['text_len'] for sample in batch)

    batch_size = len(batch)
    n_mels = batch[0]['audio_features'].size(1)

    # Initialize padded tensors
    audio_features = torch.zeros(batch_size, max_audio_len, n_mels)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)

    texts = []

    # Fill tensors
    for i, sample in enumerate(batch):
        audio_len = sample['audio_len']
        text_len = sample['text_len']

        audio_features[i, :audio_len] = sample['audio_features']
        text_tokens[i, :text_len] = sample['text_tokens']
        audio_lengths[i] = audio_len
        text_lengths[i] = text_len
        texts.append(sample['text'])

    return {
        'audio_features': audio_features,
        'text_tokens': text_tokens,
        'audio_lengths': audio_lengths,
        'text_lengths': text_lengths,
        'texts': texts
    }


def create_data_loaders(train_file, val_file, audio_dir, tokenizer,
                        batch_size=32, num_workers=4):
    """
    Create training and validation data loaders

    Args:
        train_file: Path to training data JSON
        val_file: Path to validation data JSON
        audio_dir: Directory containing audio files
        tokenizer: TextTokenizer instance
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    preprocessor = AudioPreprocessor()

    train_dataset = SpeechToTextDataset(
        train_file, audio_dir, tokenizer, preprocessor,
        max_audio_len=3000, max_text_len=500
    )

    val_dataset = SpeechToTextDataset(
        val_file, audio_dir, tokenizer, preprocessor,
        max_audio_len=3000, max_text_len=500
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = TextTokenizer()
    texts = ["hello world", "speech to text", "transformer model"]
    tokenizer.build_vocab(texts)

    print(f"Vocabulary size: {len(tokenizer)}")

    encoded = tokenizer.encode("hello")
    print(f"Encoded 'hello': {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test audio preprocessor
    preprocessor = AudioPreprocessor()
    print(
        f"Audio preprocessor initialized with {preprocessor.n_mels} mel bands")
