"""
Quick test script to verify the model and dataset components
"""

from utils import calculate_wer, LabelSmoothedCrossEntropy
from dataset import TextTokenizer, AudioPreprocessor
from model import SpeechToTextTransformer
import torch
import sys
sys.path.append('src')


def test_model():
    """Test model forward pass"""
    print("Testing SpeechToTextTransformer...")

    model = SpeechToTextTransformer(
        input_dim=80,
        d_model=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=1024,
        vocab_size=1000,
        dropout=0.1
    )

    # Test forward pass
    batch_size = 2
    src_seq_len = 100
    tgt_seq_len = 20

    src = torch.randn(batch_size, src_seq_len, 80)
    tgt = torch.randint(0, 1000, (batch_size, tgt_seq_len))

    output = model(src, tgt)

    print(f"✓ Input shape: {src.shape}")
    print(f"✓ Target shape: {tgt.shape}")
    print(f"✓ Output shape: {output.shape}")

    # Test greedy decoding
    decoded = model.greedy_decode(src, max_len=30)
    print(f"✓ Decoded shape: {decoded.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    print("Model test passed! ✓\n")


def test_tokenizer():
    """Test tokenizer"""
    print("Testing TextTokenizer...")

    tokenizer = TextTokenizer()
    texts = ["hello world", "speech to text", "transformer model"]
    tokenizer.build_vocab(texts)

    print(f"✓ Vocabulary size: {len(tokenizer)}")

    # Test encoding
    encoded = tokenizer.encode("hello world")
    print(f"✓ Encoded 'hello world': {encoded}")

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"✓ Decoded: '{decoded}'")

    assert decoded == "hello world", "Decoding mismatch!"

    print("Tokenizer test passed! ✓\n")


def test_preprocessor():
    """Test audio preprocessor"""
    print("Testing AudioPreprocessor...")

    preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=80)

    # Create dummy audio
    duration = 2.0  # seconds
    sample_rate = 16000
    waveform = torch.randn(1, int(duration * sample_rate))

    # Extract features
    features = preprocessor.extract_features(waveform)

    print(f"✓ Waveform shape: {waveform.shape}")
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Number of mel bands: {preprocessor.n_mels}")

    # Test normalization
    normalized = preprocessor.normalize(features)
    print(f"✓ Normalized mean: {normalized.mean():.4f}")
    print(f"✓ Normalized std: {normalized.std():.4f}")

    print("Audio preprocessor test passed! ✓\n")


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")

    # Test loss function
    criterion = LabelSmoothedCrossEntropy(smoothing=0.1)
    pred = torch.randn(32, 1000)
    target = torch.randint(0, 1000, (32,))
    loss = criterion(pred, target)
    print(f"✓ Label smoothed CE loss: {loss.item():.4f}")

    # Test WER
    refs = ["hello world", "this is a test"]
    hyps = ["hello world", "this is test"]
    wer = calculate_wer(refs, hyps)
    print(f"✓ WER: {wer:.4f}")

    print("Utils test passed! ✓\n")


def main():
    print("=" * 50)
    print("Speech-to-Text Transformer Test Suite")
    print("=" * 50)
    print()

    try:
        test_model()
        test_tokenizer()
        test_preprocessor()
        test_utils()

        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        print("\nYour setup is ready for training!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
