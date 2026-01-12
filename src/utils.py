"""
Utility functions for Speech-to-Text project
Includes loss functions, metrics, and helper functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jiwer import wer as calculate_word_error_rate


class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing
    Helps prevent overconfidence and improves generalization
    """

    def __init__(self, smoothing=0.1, ignore_index=-100):
        """
        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
            ignore_index: Index to ignore in loss calculation
        """
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [batch_size * seq_len, vocab_size]
            target: Target labels [batch_size * seq_len]

        Returns:
            Loss value
        """
        vocab_size = pred.size(-1)

        # Create smoothed labels
        smoothed_target = torch.zeros_like(pred).scatter_(
            1, target.unsqueeze(1), self.confidence
        )
        smoothed_target += self.smoothing / (vocab_size - 1)

        # Handle ignore index
        if self.ignore_index >= 0:
            mask = target == self.ignore_index
            smoothed_target.masked_fill_(mask.unsqueeze(1), 0)

        # Calculate KL divergence
        loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            smoothed_target,
            reduction='batchmean'
        )

        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def calculate_wer(references, hypotheses):
    """
    Calculate Word Error Rate (WER)

    Args:
        references: List of reference transcriptions
        hypotheses: List of predicted transcriptions

    Returns:
        WER as a float (0.0 to 1.0+)
    """
    try:
        # Filter out empty strings
        valid_pairs = [
            (ref, hyp) for ref, hyp in zip(references, hypotheses)
            if ref.strip() and hyp.strip()
        ]

        if not valid_pairs:
            return 1.0

        refs, hyps = zip(*valid_pairs)
        return calculate_word_error_rate(list(refs), list(hyps))
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 1.0


def calculate_cer(references, hypotheses):
    """
    Calculate Character Error Rate (CER)

    Args:
        references: List of reference transcriptions
        hypotheses: List of predicted transcriptions

    Returns:
        CER as a float (0.0 to 1.0+)
    """
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref = ref.strip()
        hyp = hyp.strip()

        if not ref:
            continue

        # Calculate Levenshtein distance at character level
        errors = levenshtein_distance(ref, hyp)
        total_errors += errors
        total_chars += len(ref)

    if total_chars == 0:
        return 1.0

    return total_errors / total_chars


def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two strings

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def count_parameters(model):
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def format_time(seconds):
    """
    Format seconds to human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_predictions(predictions, references, output_file):
    """
    Save predictions and references to a file for analysis

    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        output_file: Path to output file
    """
    import json

    results = []
    for pred, ref in zip(predictions, references):
        wer_score = calculate_wer([ref], [pred])
        cer_score = calculate_cer([ref], [pred])

        results.append({
            'reference': ref,
            'prediction': pred,
            'wer': wer_score,
            'cer': cer_score
        })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved predictions to {output_file}")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_masks(src_len, tgt_len, pad_idx=0):
    """
    Create masks for transformer

    Args:
        src_len: Source sequence lengths [batch_size]
        tgt_len: Target sequence lengths [batch_size]
        pad_idx: Padding index

    Returns:
        Tuple of (src_mask, tgt_mask)
    """
    batch_size = len(src_len)
    max_src_len = max(src_len)
    max_tgt_len = max(tgt_len)

    # Source mask (to ignore padding)
    src_mask = torch.zeros(batch_size, max_src_len)
    for i, length in enumerate(src_len):
        src_mask[i, :length] = 1

    # Target mask (causal + padding)
    tgt_mask = torch.triu(torch.ones(max_tgt_len, max_tgt_len), diagonal=1)
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))

    return src_mask, tgt_mask


def plot_training_curves(train_losses, val_losses, output_path):
    """
    Plot training and validation loss curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

        print(f"Saved training curves to {output_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot generation")


if __name__ == "__main__":
    # Test label smoothed cross entropy
    criterion = LabelSmoothedCrossEntropy(smoothing=0.1)
    pred = torch.randn(32, 5000)
    target = torch.randint(0, 5000, (32,))
    loss = criterion(pred, target)
    print(f"Label smoothed CE loss: {loss.item():.4f}")

    # Test WER calculation
    refs = ["hello world", "this is a test"]
    hyps = ["hello world", "this is test"]
    wer = calculate_wer(refs, hyps)
    print(f"WER: {wer:.4f}")

    # Test CER calculation
    cer = calculate_cer(refs, hyps)
    print(f"CER: {cer:.4f}")
