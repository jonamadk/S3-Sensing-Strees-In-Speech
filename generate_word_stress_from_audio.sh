#!/bin/bash
# Complete workflow to generate word stress features from audio dataset
# Usage: ./generate_word_stress_from_audio.sh [max_samples]

set -e

MAX_SAMPLES=${1:-""}

if [ -n "$MAX_SAMPLES" ]; then
    echo "Processing first $MAX_SAMPLES samples..."
    SAMPLES_ARG="--max-samples $MAX_SAMPLES"
    OUTPUT_SUFFIX="_${MAX_SAMPLES}samples"
else
    echo "Processing ALL samples from train.json..."
    SAMPLES_ARG=""
    OUTPUT_SUFFIX="_full"
fi

PROSODY_DIR="data/prosody${OUTPUT_SUFFIX}"
WORD_STRESS_DIR="data/word_stress${OUTPUT_SUFFIX}"

echo "=========================================="
echo "STEP 1: Extract phoneme-level features"
echo "=========================================="
python prosody_arpabet_simple.py \
  --data-json data/train.json \
  --audio-dir data_source/en/clips \
  --output-dir "$PROSODY_DIR" \
  $SAMPLES_ARG

echo ""
echo "=========================================="
echo "STEP 2: Cluster word stress levels"
echo "=========================================="
python cluster_word_stress.py \
  --input-json "${PROSODY_DIR}/phoneme_features_arpabet.json" \
  --output-dir "$WORD_STRESS_DIR" \
  --n-clusters 3 \
  --method kmeans

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo "Outputs:"
echo "  Phonemes: ${PROSODY_DIR}/phoneme_features_arpabet.json"
echo "  Word Stress: ${WORD_STRESS_DIR}/word_stress_features.json"
echo "  Visualizations: ${WORD_STRESS_DIR}/*.png"
echo ""
