#!/usr/bin/env python3
"""
Process other.tsv from Common Voice dataset to create train.json and val.json
"""
import json
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


def process_tsv_to_json(tsv_path, clips_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Process TSV file and create train.json and val.json

    Args:
        tsv_path: Path to other.tsv file
        clips_dir: Directory containing audio clips
        output_dir: Directory to save train.json and val.json
        train_ratio: Ratio for training set (default 0.8)
        val_ratio: Ratio for validation set (default 0.1)
    """
    # Read TSV file
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            clip_filename = row['path']
            sentence = row['sentence'].strip()

            # Skip empty sentences
            if not sentence:
                continue

            # Create full path to audio file for existence check
            audio_path = str(Path(clips_dir) / clip_filename)

            # Check if file exists
            if Path(audio_path).exists():
                data.append({
                    'audio': clip_filename,  # Store only filename
                    'text': sentence
                })
            else:
                print(f"Warning: File not found: {audio_path}")

    print(f"Total valid samples: {len(data)}")

    if len(data) == 0:
        print("No valid data found!")
        return

    # Split into train, val, test
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_ratio,
        random_state=42
    )

    # Second split: separate train and val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=42
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save train.json
    train_path = Path(output_dir) / 'train.json'
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {train_path}")

    # Save val.json
    val_path = Path(output_dir) / 'val.json'
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {val_path}")

    # Save test.json
    test_path = Path(output_dir) / 'test.json'
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {test_path}")

    # Print sample
    print("\nSample from train.json:")
    for i, sample in enumerate(train_data[:3]):
        print(f"{i+1}. Audio: {sample['audio']}")
        print(f"   Text: {sample['text']}")


if __name__ == '__main__':
    # Paths
    tsv_path = 'data_source/en/other.tsv'
    clips_dir = 'data_source/en/clips'
    output_dir = 'data'

    # Process
    process_tsv_to_json(tsv_path, clips_dir, output_dir)
