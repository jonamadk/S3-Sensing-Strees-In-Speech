#!/usr/bin/env python3
"""
Simple ARPAbet Phoneme Extraction using espeak subprocess
Generates the exact format you requested with proper phoneme representations.
"""

import argparse
import json
import os
import subprocess
import librosa
import numpy as np
import parselmouth
from tqdm import tqdm


# IPA to ARPAbet mapping
IPA2ARPABET = {
    # Diphthongs (check 2-char first)
    "aʊ": "AW", "aɪ": "AY", "eɪ": "EY", "oʊ": "OW", "ɔɪ": "OY",
    # Vowels
    "ɑ": "AA", "ɑː": "AA", "æ": "AE", "ʌ": "AH", "ə": "AH0", "ɔ": "AO", "ɔː": "AO",
    "ɛ": "EH", "ɝ": "ER", "ɚ": "ER0", "ɪ": "IH", "ɨ": "IH0", "ᵻ": "IH0",
    "i": "IY", "iː": "IY", "ʊ": "UH", "u": "UW", "uː": "UW",
    # Consonants (digraphs)
    "tʃ": "CH", "dʒ": "JH", "ð": "DH", "θ": "TH", "ʃ": "SH", "ʒ": "ZH", "ŋ": "NG",
    # Regular consonants
    "b": "B", "d": "D", "f": "F", "ɡ": "G", "g": "G", "h": "HH", "k": "K",
    "l": "L", "m": "M", "n": "N", "p": "P", "ɹ": "R", "r": "R", "s": "S",
    "t": "T", "v": "V", "w": "W", "j": "Y", "z": "Z", "ɾ": "D",
}


def phonemize_word(word: str) -> str:
    """Get IPA phonemes for a word using espeak."""
    try:
        result = subprocess.run(
            ['/opt/homebrew/bin/espeak', '-q', '--ipa', word],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return ""


def ipa_to_arpabet(ipa: str) -> list:
    """Convert IPA string to ARPAbet phoneme list."""
    phonemes = []
    i = 0
    ipa = ipa.strip()

    while i < len(ipa):
        # Skip stress markers and punctuation
        if ipa[i] in 'ˈˌːˑ .,!?;:-':
            i += 1
            continue

        # Try 2-char match
        if i + 1 < len(ipa) and ipa[i:i+2] in IPA2ARPABET:
            phonemes.append(IPA2ARPABET[ipa[i:i+2]])
            i += 2
        # Try 1-char match
        elif ipa[i] in IPA2ARPABET:
            phonemes.append(IPA2ARPABET[ipa[i]])
            i += 1
        else:
            # Unknown - store for debugging
            phonemes.append(f"?{ipa[i]}?")
            i += 1

    return phonemes


def extract_pitch(audio_path: str, start: float, end: float) -> float:
    """Extract pitch using Praat."""
    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch(time_step=0.01)
        values = []
        t = start
        while t <= end:
            f0 = pitch.get_value_at_time(t)
            if f0 and 75 <= f0 <= 600:
                values.append(f0)
            t += 0.01
        return float(np.mean(values)) if values else 0.0
    except:
        return 0.0


def extract_energy(audio_path: str, start: float, end: float) -> float:
    """Extract RMS energy."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        if len(segment) == 0:
            return 0.0
        rms = librosa.feature.rms(y=segment)[0]
        return float(np.mean(rms))
    except:
        return 0.0


def process_sample(audio_path: str, text: str, sample_id: int) -> list:
    """Process one audio sample and extract phoneme-level features."""

    # Get audio duration
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        total_duration = len(y) / sr
    except:
        return []

    words = text.split()
    if not words:
        return []

    phoneme_features = []
    word_duration = total_duration / len(words)

    for word_idx, word in enumerate(words):
        word_start = word_idx * word_duration
        word_end = (word_idx + 1) * word_duration

        # Get IPA phonemes
        ipa = phonemize_word(word)
        if not ipa:
            continue

        # Convert to ARPAbet
        arpabet_phonemes = ipa_to_arpabet(ipa)
        if not arpabet_phonemes:
            continue

        # Distribute phonemes across word duration
        phoneme_duration = (word_end - word_start) / len(arpabet_phonemes)

        for phon_idx, phoneme in enumerate(arpabet_phonemes):
            phon_start = word_start + (phon_idx * phoneme_duration)
            phon_end = phon_start + phoneme_duration

            pitch = extract_pitch(audio_path, phon_start, phon_end)
            energy = extract_energy(audio_path, phon_start, phon_end)

            # Output format matching your example
            phoneme_features.append({
                'char': phoneme,
                'start': round(phon_start, 3),
                'end': round(phon_end, 3),
                'word': word,
                'pitch': round(pitch, 2),
                'energy': round(energy, 4),
                'sentence': text
            })

    return phoneme_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-json', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    # Load data
    with open(args.data_json, 'r') as f:
        data = json.load(f)

    if args.max_samples:
        data = data[:args.max_samples]

    # Process samples
    all_phonemes = []

    print(f"\nProcessing {len(data)} samples...")
    for idx, sample in enumerate(tqdm(data)):
        audio_file = os.path.join(args.audio_dir, sample['audio'])
        if not os.path.exists(audio_file):
            continue

        phonemes = process_sample(audio_file, sample['text'], idx)
        all_phonemes.extend(phonemes)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 'phoneme_features_arpabet.json')

    with open(output_file, 'w') as f:
        json.dump(all_phonemes, f, indent=2)

    # Stats
    print("\n" + "="*70)
    print(f"Total phonemes: {len(all_phonemes)}")
    unique = set(p['char'] for p in all_phonemes)
    print(f"Unique phonemes: {len(unique)}")
    print(f"Phoneme inventory: {sorted(unique)}")

    if all_phonemes:
        print(f"\nSample output (first 10):")
        for p in all_phonemes[:10]:
            print(f"  {p}")

    print(f"\nSaved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
