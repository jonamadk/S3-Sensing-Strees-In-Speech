#!/usr/bin/env python3
"""
Fix unknown IPA symbols in existing phoneme and word stress data.
Replaces ?symbol? patterns with proper ARPAbet equivalents.
"""

import json
import argparse
import os
import shutil

# Mapping of IPA symbols to ARPAbet
SYMBOL_FIXES = {
    "?ɒ?": "AO",   # British "lot" vowel (hot, dog)
    "?a?": "AE",   # Open front vowel (cat, bat)
    "?ɜ?": "ER",   # "Nurse" vowel (bird, her)
    "?ɐ?": "AH0",  # Near-open central vowel (unstressed schwa-like)
    "?e?": "EY",   # Close-mid front vowel (they, say)
    "?o?": "OW",   # Close-mid back vowel (go, no)
    "?ᵻ?": "IH0",  # Close central unrounded vowel
}


def fix_phoneme_data(input_file, output_file):
    """Fix unknown symbols in phoneme-level data."""
    print(f"Loading phoneme data from: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Count fixes
    fix_count = 0
    for phoneme in data:
        char = phoneme['char']
        if char in SYMBOL_FIXES:
            phoneme['char'] = SYMBOL_FIXES[char]
            fix_count += 1

    print(f"Fixed {fix_count} unknown symbols in {len(data)} phonemes")

    # Backup original
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"Created backup: {backup_file}")

    # Save fixed version
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved fixed data to: {output_file}")

    # Show unique phonemes
    unique = sorted(set(p['char'] for p in data))
    print(f"\nUnique phonemes after fix: {len(unique)}")
    print(f"Phoneme inventory: {unique}")

    # Check for remaining unknowns
    remaining = [p for p in data if '?' in p['char']]
    if remaining:
        print(f"\n⚠ WARNING: {len(remaining)} unknown symbols still remain:")
        unknown_chars = set(p['char'] for p in remaining)
        for char in sorted(unknown_chars):
            count = sum(1 for p in data if p['char'] == char)
            print(f"  {char} ({count} times)")
    else:
        print("\n✓ No unknown symbols remaining!")


def fix_word_stress_data(input_file, output_file):
    """Fix unknown symbols in word stress data."""
    print(f"\nLoading word stress data from: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Count fixes
    fix_count = 0
    for word_data in data:
        phonemes = word_data.get('phonemes', '')
        original = phonemes

        # Replace all unknown symbols
        for unknown, fixed in SYMBOL_FIXES.items():
            if unknown in phonemes:
                phonemes = phonemes.replace(unknown, fixed)
                fix_count += 1

        if phonemes != original:
            word_data['phonemes'] = phonemes

    print(f"Fixed {fix_count} unknown symbols in {len(data)} words")

    # Backup original
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"Created backup: {backup_file}")

    # Save fixed version
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved fixed data to: {output_file}")

    # Check for remaining unknowns
    remaining = [w for w in data if '?' in w.get('phonemes', '')]
    if remaining:
        print(
            f"\n⚠ WARNING: {len(remaining)} words still have unknown symbols:")
        import re
        unknown_patterns = set()
        for w in remaining:
            unknowns = re.findall(r'\?[^?]+\?', w['phonemes'])
            unknown_patterns.update(unknowns)
        for pattern in sorted(unknown_patterns):
            count = sum(1 for w in data if pattern in w.get('phonemes', ''))
            print(f"  {pattern} (in {count} words)")
    else:
        print("\n✓ No unknown symbols remaining!")


def fix_csv_data(input_file, output_file):
    """Fix unknown symbols in CSV word stress data."""
    import pandas as pd

    print(f"\nLoading CSV data from: {input_file}")
    df = pd.read_csv(input_file)

    # Count fixes
    fix_count = 0
    if 'phonemes' in df.columns:
        for idx, phonemes in enumerate(df['phonemes']):
            if pd.isna(phonemes):
                continue
            original = phonemes
            for unknown, fixed in SYMBOL_FIXES.items():
                if unknown in phonemes:
                    phonemes = phonemes.replace(unknown, fixed)
                    fix_count += 1
            if phonemes != original:
                df.at[idx, 'phonemes'] = phonemes

    print(f"Fixed {fix_count} unknown symbols in {len(df)} rows")

    # Backup original
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"Created backup: {backup_file}")

    # Save fixed version
    df.to_csv(output_file, index=False)
    print(f"Saved fixed CSV to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fix unknown IPA symbols in phoneme/word stress data')
    parser.add_argument('--phoneme-json',
                        help='Path to phoneme_features_arpabet.json')
    parser.add_argument('--word-json',
                        help='Path to word_stress_features.json')
    parser.add_argument('--word-csv',
                        help='Path to word_stress_features.csv')
    parser.add_argument('--all', action='store_true',
                        help='Fix all default files')

    args = parser.parse_args()

    if args.all:
        # Fix all default files
        print("="*80)
        print("FIXING ALL DEFAULT DATA FILES")
        print("="*80)

        phoneme_file = 'data/prosody_arpabet_full/phoneme_features_arpabet.json'
        word_json = 'data/word_stress_clustered/word_stress_features.json'
        word_csv = 'data/word_stress_clustered/word_stress_features.csv'

        if os.path.exists(phoneme_file):
            fix_phoneme_data(phoneme_file, phoneme_file)

        if os.path.exists(word_json):
            fix_word_stress_data(word_json, word_json)

        if os.path.exists(word_csv):
            fix_csv_data(word_csv, word_csv)

        # Also fix binary clustering data
        word_json_binary = 'data/word_stress_binary/word_stress_features.json'
        word_csv_binary = 'data/word_stress_binary/word_stress_features.csv'

        if os.path.exists(word_json_binary):
            fix_word_stress_data(word_json_binary, word_json_binary)

        if os.path.exists(word_csv_binary):
            fix_csv_data(word_csv_binary, word_csv_binary)

    else:
        if args.phoneme_json:
            fix_phoneme_data(args.phoneme_json, args.phoneme_json)

        if args.word_json:
            fix_word_stress_data(args.word_json, args.word_json)

        if args.word_csv:
            fix_csv_data(args.word_csv, args.word_csv)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
