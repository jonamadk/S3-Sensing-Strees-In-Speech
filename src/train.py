"""
Training script for Speech-to-Text Transformer
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from model import SpeechToTextTransformer
from dataset import TextTokenizer, create_data_loaders
from utils import LabelSmoothedCrossEntropy, calculate_wer, EarlyStopping


class Trainer:
    """Trainer class for Speech-to-Text model"""

    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        """
        Initialize trainer

        Args:
            model: SpeechToTextTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Text tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Loss function with label smoothing
        self.criterion = LabelSmoothedCrossEntropy(
            smoothing=config['training'].get('label_smoothing', 0.1),
            ignore_index=tokenizer.char2idx['<PAD>']
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Mixed precision training
        self.use_amp = config['training'].get(
            'use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping'].get('patience', 10),
            min_delta=config['early_stopping'].get('min_delta', 0.001)
        )

        # Tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Checkpointing
        self.checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio_features = batch['audio_features'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)

            # Prepare inputs and targets
            tgt_input = text_tokens[:, :-1]  # Remove last token
            tgt_output = text_tokens[:, 1:]  # Remove first token

            # Generate target mask
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(
                        audio_features, tgt_input, tgt_mask=tgt_mask)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1)
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    audio_features, tgt_input, tgt_mask=tgt_mask)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                             'avg_loss': f'{avg_loss:.4f}'})

        return total_loss / num_batches

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")

            for batch in pbar:
                # Move data to device
                audio_features = batch['audio_features'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)

                # Prepare inputs and targets
                tgt_input = text_tokens[:, :-1]
                tgt_output = text_tokens[:, 1:]

                # Generate target mask
                tgt_mask = self.model.generate_square_subsequent_mask(
                    tgt_input.size(1)
                ).to(self.device)

                # Forward pass
                logits = self.model(
                    audio_features, tgt_input, tgt_mask=tgt_mask)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )

                total_loss += loss.item()

                # Generate predictions for WER calculation
                predictions = self.model.greedy_decode(
                    audio_features,
                    max_len=100,
                    start_token=self.tokenizer.char2idx['<SOS>'],
                    end_token=self.tokenizer.char2idx['<EOS>']
                )

                # Decode predictions and references
                for pred, ref in zip(predictions, batch['texts']):
                    pred_text = self.tokenizer.decode(pred)
                    all_predictions.append(pred_text)
                    all_references.append(ref)

        avg_loss = total_loss / len(self.val_loader)

        # Calculate WER (Word Error Rate)
        wer = calculate_wer(all_references, all_predictions)

        return avg_loss, wer

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total training samples: {len(self.train_loader.dataset)}")
        print(f"Total validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.start_epoch, num_epochs):
            start_time = time.time()

            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, wer = self.validate(epoch)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Timing
            epoch_time = time.time() - start_time

            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"WER: {wer:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Speech-to-Text Transformer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    print("Configuration:")
    print(json.dumps(config, indent=2))

    # Initialize tokenizer
    tokenizer = TextTokenizer()

    # Build vocabulary if needed
    vocab_file = config['data']['vocab_file']
    if not os.path.exists(vocab_file):
        print("Building vocabulary...")
        # Load all training texts
        with open(config['data']['train_file'], 'r') as f:
            train_data = json.load(f)
        texts = [sample['text'] for sample in train_data]
        tokenizer.build_vocab(texts)
        tokenizer.save_vocab(vocab_file)
        print(f"Vocabulary saved to {vocab_file}")
    else:
        tokenizer.load_vocab(vocab_file)
        print(f"Vocabulary loaded from {vocab_file}")

    print(f"Vocabulary size: {len(tokenizer)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        config['data']['train_file'],
        config['data']['val_file'],
        config['data']['audio_dir'],
        tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )

    # Initialize model
    model = SpeechToTextTransformer(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        vocab_size=len(tokenizer),
        max_seq_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout']
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train model
    trainer.train(config['training']['num_epochs'])


if __name__ == "__main__":
    main()
