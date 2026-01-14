#!/usr/bin/env python3
"""
Attention-based Neural Networks for Word Stress Prediction

This script trains attention-based deep learning models to predict word stress
from prosodic features using PyTorch. Implements:
- Multi-Head Self-Attention
- Feature-wise Attention
- Transformer Encoder
- Attention visualization
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class StressDataset(Dataset):
    """PyTorch Dataset for word stress features."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureAttention(nn.Module):
    """Feature-wise attention mechanism."""

    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        attention_weights = self.attention(x)  # (batch_size, input_dim)
        weighted_features = x * attention_weights  # Element-wise multiplication
        return weighted_features, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for feature relationships."""

    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]

        # Add sequence dimension for attention computation
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        Q = self.query(x)  # (batch_size, 1, input_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        # (batch_size, num_heads, 1, head_dim)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(
            batch_size, 1, self.input_dim)
        out = self.fc_out(out)

        return out.squeeze(1), attention.mean(dim=1).squeeze(1)


class AttentionStressClassifier(nn.Module):
    """Attention-based classifier with feature attention."""

    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.3):
        super(AttentionStressClassifier, self).__init__()

        self.feature_attention = FeatureAttention(input_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # Apply feature attention
        x, attention_weights = self.feature_attention(x)

        # Feed-forward layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x, attention_weights


class MultiHeadStressClassifier(nn.Module):
    """Classifier with multi-head self-attention."""

    def __init__(self, input_dim, num_classes, num_heads=4, hidden_dim=128, dropout=0.3):
        super(MultiHeadStressClassifier, self).__init__()

        # Adjust input_dim to be divisible by num_heads
        self.proj_dim = (input_dim // num_heads) * num_heads
        if self.proj_dim != input_dim:
            self.input_projection = nn.Linear(input_dim, self.proj_dim)
        else:
            self.input_projection = None

        self.multi_head_attention = MultiHeadAttention(
            self.proj_dim, num_heads)

        self.fc1 = nn.Linear(self.proj_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Apply multi-head attention
        x, attention_weights = self.multi_head_attention(x)

        # Feed-forward layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x, attention_weights


class TransformerStressClassifier(nn.Module):
    """Transformer encoder-based classifier."""

    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128, dropout=0.3):
        super(TransformerStressClassifier, self).__init__()

        # Adjust input_dim to be divisible by num_heads
        self.proj_dim = (input_dim // num_heads) * num_heads
        if self.proj_dim != input_dim:
            self.input_projection = nn.Linear(input_dim, self.proj_dim)
        else:
            self.input_projection = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.proj_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(self.proj_dim, num_classes)

    def forward(self, x):
        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, proj_dim)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, proj_dim)

        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, proj_dim)

        # Final classification
        x = self.fc(x)

        return x, None  # No attention weights to visualize


def load_data(json_path):
    """Load word stress features from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} words from {json_path}")
    return df


def prepare_features(df):
    """Prepare features and labels."""
    feature_cols = [
        'word_duration', 'vowel_duration', 'consonant_duration', 'vowel_ratio',
        'num_vowels', 'num_phonemes',
        'pitch_mean', 'pitch_max', 'pitch_min', 'pitch_range', 'pitch_std',
        'pitch_slope', 'pitch_madiff',
        'vowel_pitch_mean', 'vowel_pitch_max',
        'energy_mean', 'energy_max',
        'pre_pause', 'post_pause',
        'pos_norm_start', 'pos_norm_center'
    ]

    available_features = [col for col in feature_cols if col in df.columns]

    X = df[available_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df['stress_label'].copy()

    # Detect classification type
    unique_labels = y.unique()
    if 'stressed' in unique_labels and 'unstressed' in unique_labels:
        classification_type = 'binary'
        label_mapping = {'unstressed': 0, 'stressed': 1}
        class_names = ['Unstressed', 'Stressed']
    else:
        classification_type = '3class'
        label_mapping = {'unstressed': 0, 'secondary': 1, 'primary': 2}
        class_names = ['Unstressed', 'Secondary', 'Primary']

    y = y.map(label_mapping)

    print(f"\nClassification type: {classification_type}")
    print(f"Features: {len(available_features)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    return X, y, available_features, class_names, classification_type


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attention_weights = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs, attention_weights = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            if attention_weights is not None:
                all_attention_weights.append(attention_weights.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    if all_attention_weights:
        all_attention_weights = np.vstack(all_attention_weights)
    else:
        all_attention_weights = None

    return avg_loss, accuracy, f1, all_preds, all_labels, all_attention_weights


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, model_name):
    """Train model with validation."""
    print(f"\nTraining {model_name}...")

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_f1 = 0
    best_model_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return model, history


def visualize_attention_weights(attention_weights, feature_names, output_path, model_name):
    """Visualize average attention weights across features."""
    avg_attention = attention_weights.mean(axis=0)

    # Ensure dimensions match
    if len(avg_attention) != len(feature_names):
        print(
            f"Warning: Attention weights shape {avg_attention.shape} doesn't match features {len(feature_names)}")
        # Truncate or pad as needed
        if len(avg_attention) > len(feature_names):
            avg_attention = avg_attention[:len(feature_names)]
        else:
            # Pad with zeros
            padding = np.zeros(len(feature_names) - len(avg_attention))
            avg_attention = np.concatenate([avg_attention, padding])

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_names)), avg_attention)
    plt.xticks(range(len(feature_names)),
               feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Average Attention Weight')
    plt.title(f'{model_name} - Feature Importance via Attention Weights')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create sorted importance table
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Attention Weight': avg_attention
    }).sort_values('Attention Weight', ascending=False)

    return importance_df


def plot_training_history(histories, output_dir):
    """Plot training history for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for model_name, history in histories.items():
        # Loss
        axes[0, 0].plot(history['train_loss'],
                        label=f'{model_name} - Train', alpha=0.7)
        axes[0, 1].plot(history['val_loss'],
                        label=f'{model_name} - Val', alpha=0.7)

        # Accuracy
        axes[1, 0].plot(history['train_acc'],
                        label=f'{model_name} - Train', alpha=0.7)
        axes[1, 1].plot(history['val_acc'],
                        label=f'{model_name} - Val', alpha=0.7)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'attention_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results, class_names, output_dir):
    """Plot confusion matrices for all models."""
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))

    if num_models == 1:
        axes = [axes]

    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_true'], result['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx])
        axes[idx].set_title(f'{model_name}\nF1: {result["f1"]:.4f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'attention_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results, histories, class_names, output_dir):
    """Generate comprehensive markdown report."""
    report_path = os.path.join(output_dir, 'ATTENTION_MODELS_REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# Attention-Based Models for Word Stress Prediction\n\n")
        f.write("## Model Comparison\n\n")

        # Results table
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")

        for model_name, result in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
            f.write(f"| {model_name} | {result['accuracy']:.4f} | {result['precision']:.4f} | "
                    f"{result['recall']:.4f} | {result['f1']:.4f} |\n")

        f.write("\n## Detailed Results\n\n")

        for model_name, result in results.items():
            f.write(f"### {model_name}\n\n")
            f.write("```\n")
            f.write(result['classification_report'])
            f.write("```\n\n")

        f.write("\n## Training Summary\n\n")
        f.write("- **Architecture**: Attention-based neural networks\n")
        f.write("- **Attention Types**: Feature-wise, Multi-head, Transformer\n")
        f.write("- **Framework**: PyTorch\n")
        f.write("- **Early Stopping**: Patience = 20 epochs\n")
        f.write("- **Optimization**: Adam optimizer\n\n")

        f.write("## Key Findings\n\n")
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        f.write(
            f"- **Best Model**: {best_model[0]} (F1: {best_model[1]['f1']:.4f})\n")
        f.write(f"- **Classes**: {', '.join(class_names)}\n")
        f.write(
            "- **Attention Mechanism**: Learns feature importance automatically\n\n")

    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train attention-based models for word stress prediction')
    parser.add_argument('--input-json', required=True,
                        help='Path to word_stress_features.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for models and results')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--random-state', type=int,
                        default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\n" + "="*80)
    print("ATTENTION-BASED MODELS FOR WORD STRESS PREDICTION")
    print("="*80)

    # Load and prepare data
    df = load_data(args.input_json)
    X, y, feature_names, class_names, classification_type = prepare_features(
        df)
    num_classes = len(class_names)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets
    train_dataset = StressDataset(X_train_scaled, y_train.values)
    test_dataset = StressDataset(X_test_scaled, y_test.values)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train_scaled.shape[1]

    # Define models
    models_config = {
        'Feature Attention': AttentionStressClassifier(input_dim, num_classes, hidden_dim=128, dropout=0.3),
        'Multi-Head Attention': MultiHeadStressClassifier(input_dim, num_classes, num_heads=4, hidden_dim=128, dropout=0.3),
        'Transformer': TransformerStressClassifier(input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128, dropout=0.3)
    }

    results = {}
    histories = {}
    attention_weights_dict = {}

    criterion = nn.CrossEntropyLoss()

    # Train each model
    for model_name, model in models_config.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train
        model, history = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            args.epochs, device, model_name
        )

        histories[model_name] = history

        # Evaluate on test set
        _, accuracy, f1, y_pred, y_true, attention_weights = evaluate(
            model, test_loader, criterion, device
        )

        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        report = classification_report(
            y_true, y_pred, target_names=class_names)

        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'classification_report': report
        }

        # Save attention weights
        if attention_weights is not None:
            attention_weights_dict[model_name] = attention_weights
            importance_df = visualize_attention_weights(
                attention_weights, feature_names,
                os.path.join(
                    args.output_dir, f'{model_name.lower().replace(" ", "_")}_attention.png'),
                model_name
            )
            importance_df.to_csv(
                os.path.join(
                    args.output_dir, f'{model_name.lower().replace(" ", "_")}_importance.csv'),
                index=False
            )

        # Save model
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, f'{model_name.lower().replace(" ", "_")}_model.pt'))

        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")

    # Save scaler
    import pickle
    with open(os.path.join(args.output_dir, 'attention_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Create visualizations
    plot_training_history(histories, args.output_dir)
    plot_confusion_matrices(results, class_names, args.output_dir)

    # Generate report
    generate_report(results, histories, class_names, args.output_dir)

    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1'] for r in results.values()]
    }).sort_values('F1-Score', ascending=False)

    results_df.to_csv(os.path.join(
        args.output_dir, 'attention_results.csv'), index=False)

    print("\n" + "="*80)
    print("ATTENTION MODEL TRAINING COMPLETE!")
    print("="*80)

    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
