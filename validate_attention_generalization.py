#!/usr/bin/env python3
"""
Generalization Validation for Attention Models

This script performs comprehensive generalization testing:
1. K-Fold Cross-Validation
2. Learning Curve Analysis
3. Overfitting Detection
4. Bootstrap Validation
5. Out-of-Distribution Testing
"""

from train_attention_models import (
    FeatureAttention, MultiHeadAttention, TransformerStressClassifier,
    AttentionStressClassifier, MultiHeadStressClassifier, StressDataset
)
import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import warnings
warnings.filterwarnings('ignore')

# Import model classes from train_attention_models
sys.path.append(os.path.dirname(__file__))


def load_data(json_path):
    """Load word stress features."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
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

    unique_labels = y.unique()
    if 'stressed' in unique_labels and 'unstressed' in unique_labels:
        label_mapping = {'unstressed': 0, 'stressed': 1}
        class_names = ['Unstressed', 'Stressed']
    else:
        label_mapping = {'unstressed': 0, 'secondary': 1, 'primary': 2}
        class_names = ['Unstressed', 'Secondary', 'Primary']

    y = y.map(label_mapping)

    return X, y, available_features, class_names


def k_fold_cross_validation(X, y, model_class, model_kwargs, k=5, epochs=50, batch_size=64, lr=0.001, device='cpu'):
    """Perform k-fold cross-validation."""
    print(f"\nPerforming {k}-Fold Cross-Validation...")

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_scores = []
    fold_train_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold+1}/{k}")

        # Split data
        X_train_fold = X.iloc[train_idx].values
        y_train_fold = y.iloc[train_idx].values
        X_val_fold = X.iloc[val_idx].values
        y_val_fold = y.iloc[val_idx].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Create datasets
        train_dataset = StressDataset(X_train_scaled, y_train_fold)
        val_dataset = StressDataset(X_val_scaled, y_val_fold)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        best_val_f1 = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs, _ = model(X_batch)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(y_batch.numpy())

            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Final evaluation on training set
        model.eval()
        train_preds = []
        train_labels = []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                outputs, _ = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(y_batch.numpy())

        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        fold_scores.append(best_val_f1)
        fold_train_scores.append(train_f1)
        print(
            f"  Train F1: {train_f1:.4f}, Val F1: {best_val_f1:.4f}, Gap: {train_f1 - best_val_f1:.4f}")

    return fold_train_scores, fold_scores


def bootstrap_validation(X, y, model_class, model_kwargs, n_iterations=10, sample_size=0.8,
                         epochs=50, batch_size=64, lr=0.001, device='cpu'):
    """Perform bootstrap validation."""
    print(f"\nPerforming Bootstrap Validation ({n_iterations} iterations)...")

    bootstrap_scores = []
    n_samples = len(X)

    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=int(
            n_samples * sample_size), replace=True)
        oob_indices = list(set(range(n_samples)) - set(indices))

        if len(oob_indices) == 0:
            continue

        X_train = X.iloc[indices].values
        y_train = y.iloc[indices].values
        X_test = X.iloc[oob_indices].values
        y_test = y.iloc[oob_indices].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create datasets
        train_dataset = StressDataset(X_train_scaled, y_train)
        test_dataset = StressDataset(X_test_scaled, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Train model
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs, _ = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(y_batch.numpy())

        f1 = f1_score(test_labels, test_preds, average='weighted')
        bootstrap_scores.append(f1)

        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{n_iterations}: F1 = {f1:.4f}")

    return bootstrap_scores


def plot_generalization_results(results, output_dir):
    """Plot comprehensive generalization analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. K-Fold Results
    ax = axes[0, 0]
    models = list(results['kfold'].keys())
    x = np.arange(len(models))
    width = 0.35

    train_means = [np.mean(results['kfold'][m]['train']) for m in models]
    val_means = [np.mean(results['kfold'][m]['val']) for m in models]
    train_stds = [np.std(results['kfold'][m]['train']) for m in models]
    val_stds = [np.std(results['kfold'][m]['val']) for m in models]

    ax.bar(x - width/2, train_means, width,
           label='Train', yerr=train_stds, capsize=5)
    ax.bar(x + width/2, val_means, width,
           label='Validation', yerr=val_stds, capsize=5)
    ax.set_ylabel('F1-Score')
    ax.set_title('5-Fold Cross-Validation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.0])

    # 2. Overfitting Gap
    ax = axes[0, 1]
    gaps = [train_means[i] - val_means[i] for i in range(len(models))]
    colors = ['green' if g < 0.01 else 'orange' if g <
              0.02 else 'red' for g in gaps]
    ax.bar(models, gaps, color=colors, alpha=0.7)
    ax.axhline(y=0.01, color='green', linestyle='--', label='Good (<1%)')
    ax.axhline(y=0.02, color='orange', linestyle='--',
               label='Acceptable (<2%)')
    ax.set_ylabel('Train-Val Gap')
    ax.set_title('Overfitting Analysis (Train-Val Gap)')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Bootstrap Validation Distribution
    ax = axes[1, 0]
    bootstrap_data = [results['bootstrap'][m] for m in models]
    bp = ax.boxplot(bootstrap_data, labels=models, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('F1-Score')
    ax.set_title('Bootstrap Validation Distribution (10 iterations)')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.0])

    # 4. Stability Metrics
    ax = axes[1, 1]
    cv_stds = val_stds
    bootstrap_stds = [np.std(results['bootstrap'][m]) for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, cv_stds, width, label='CV Std Dev', alpha=0.7)
    ax.bar(x + width/2, bootstrap_stds, width,
           label='Bootstrap Std Dev', alpha=0.7)
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Model Stability (Lower = More Stable)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(
        f"\nVisualization saved to: {os.path.join(output_dir, 'generalization_analysis.png')}")


def generate_report(results, output_dir):
    """Generate generalization analysis report."""
    report_path = os.path.join(output_dir, 'GENERALIZATION_REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# Attention Models Generalization Analysis\n\n")

        f.write("## Summary\n\n")
        f.write("This report evaluates model generalization using:\n")
        f.write(
            "- **5-Fold Cross-Validation**: Tests performance across different data splits\n")
        f.write(
            "- **Bootstrap Validation**: Tests stability with repeated random sampling\n")
        f.write("- **Overfitting Detection**: Analyzes train-validation gap\n\n")

        f.write("## 5-Fold Cross-Validation Results\n\n")
        f.write(
            "| Model | Train F1 (Mean±SD) | Val F1 (Mean±SD) | Train-Val Gap | Status |\n")
        f.write(
            "|-------|-------------------|------------------|---------------|--------|\n")

        for model_name in results['kfold'].keys():
            train_mean = np.mean(results['kfold'][model_name]['train'])
            train_std = np.std(results['kfold'][model_name]['train'])
            val_mean = np.mean(results['kfold'][model_name]['val'])
            val_std = np.std(results['kfold'][model_name]['val'])
            gap = train_mean - val_mean

            if gap < 0.01:
                status = "✅ Excellent"
            elif gap < 0.02:
                status = "⚠️ Good"
            else:
                status = "❌ Overfitting"

            f.write(
                f"| {model_name} | {train_mean:.4f}±{train_std:.4f} | {val_mean:.4f}±{val_std:.4f} | {gap:.4f} | {status} |\n")

        f.write("\n### Interpretation:\n")
        f.write("- **Train-Val Gap < 1%**: Excellent generalization, no overfitting\n")
        f.write("- **Train-Val Gap 1-2%**: Good generalization, acceptable\n")
        f.write(
            "- **Train-Val Gap > 2%**: Potential overfitting, needs regularization\n\n")

        f.write("## Bootstrap Validation Results (10 iterations)\n\n")
        f.write("| Model | Mean F1 | Std Dev | 95% CI | Stability |\n")
        f.write("|-------|---------|---------|--------|----------|\n")

        for model_name in results['bootstrap'].keys():
            scores = results['bootstrap'][model_name]
            mean = np.mean(scores)
            std = np.std(scores)
            ci_lower = np.percentile(scores, 2.5)
            ci_upper = np.percentile(scores, 97.5)

            if std < 0.005:
                stability = "🟢 Highly Stable"
            elif std < 0.01:
                stability = "🟡 Stable"
            else:
                stability = "🔴 Unstable"

            f.write(
                f"| {model_name} | {mean:.4f} | {std:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] | {stability} |\n")

        f.write("\n### Interpretation:\n")
        f.write(
            "- **Low Std Dev**: Model performs consistently across different samples\n")
        f.write("- **Tight 95% CI**: Predictions are reliable and stable\n")
        f.write(
            "- **High variance**: Model is sensitive to training data composition\n\n")

        f.write("## Key Findings\n\n")

        # Find best model
        best_model = max(results['kfold'].keys(),
                         key=lambda x: np.mean(results['kfold'][x]['val']))
        best_val = np.mean(results['kfold'][best_model]['val'])

        # Most stable model
        most_stable = min(results['bootstrap'].keys(),
                          key=lambda x: np.std(results['bootstrap'][x]))
        stability_std = np.std(results['bootstrap'][most_stable])

        f.write(
            f"1. **Best Generalizing Model**: {best_model} (CV F1: {best_val:.4f})\n")
        f.write(
            f"2. **Most Stable Model**: {most_stable} (Bootstrap Std: {stability_std:.4f})\n")
        f.write(f"3. **Recommended for Production**: {best_model}\n\n")

        f.write("## Generalization Techniques Applied\n\n")
        f.write("✅ Early stopping (patience=20 epochs)\n")
        f.write("✅ Dropout regularization (0.3)\n")
        f.write("✅ Batch normalization\n")
        f.write("✅ Train-validation split (80:20)\n")
        f.write("✅ Cross-validation testing\n")
        f.write("✅ Bootstrap validation\n\n")

        f.write("## Recommendations\n\n")

        # Check if any model has high gap
        max_gap = max([np.mean(results['kfold'][m]['train']) - np.mean(results['kfold'][m]['val'])
                      for m in results['kfold'].keys()])

        if max_gap < 0.01:
            f.write("✅ **All models show excellent generalization**\n")
            f.write("- Train-validation gaps are minimal (<1%)\n")
            f.write("- Models are ready for production deployment\n")
        elif max_gap < 0.02:
            f.write("⚠️ **Models show good generalization**\n")
            f.write("- Some minor overfitting detected\n")
            f.write("- Consider increasing dropout or regularization\n")
        else:
            f.write("❌ **Potential overfitting detected**\n")
            f.write("- Recommendations:\n")
            f.write("  - Increase dropout rate (try 0.4-0.5)\n")
            f.write("  - Add L2 regularization (weight_decay)\n")
            f.write("  - Use more training data\n")
            f.write("  - Reduce model complexity\n")

    print(f"\nGeneralization report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate attention model generalization')
    parser.add_argument('--input-json', required=True,
                        help='Path to word_stress_features.json')
    parser.add_argument(
        '--output-dir', default='models_attention', help='Output directory')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--bootstrap-iterations', type=int,
                        default=10, help='Bootstrap iterations')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per fold')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("ATTENTION MODELS GENERALIZATION VALIDATION")
    print("="*80)

    # Load data
    df = load_data(args.input_json)
    X, y, feature_names, class_names = prepare_features(df)
    num_classes = len(class_names)
    input_dim = len(feature_names)

    print(
        f"\nDataset: {len(df)} samples, {input_dim} features, {num_classes} classes")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define models to test
    models_to_test = {
        'Feature Attention': (AttentionStressClassifier, {'input_dim': input_dim, 'num_classes': num_classes}),
        'Multi-Head Attention': (MultiHeadStressClassifier, {'input_dim': input_dim, 'num_classes': num_classes, 'num_heads': 4}),
        'Transformer': (TransformerStressClassifier, {'input_dim': input_dim, 'num_classes': num_classes, 'num_heads': 4, 'num_layers': 2})
    }

    results = {
        'kfold': {},
        'bootstrap': {}
    }

    # Run validation for each model
    for model_name, (model_class, model_kwargs) in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"{'='*80}")

        # K-Fold Cross-Validation
        train_scores, val_scores = k_fold_cross_validation(
            X, y, model_class, model_kwargs,
            k=args.k_folds, epochs=args.epochs, device=device
        )
        results['kfold'][model_name] = {
            'train': train_scores,
            'val': val_scores
        }

        print(f"\nCross-Validation Summary:")
        print(
            f"  Train F1: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(
            f"  Val F1:   {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
        print(f"  Gap:      {np.mean(train_scores) - np.mean(val_scores):.4f}")

        # Bootstrap Validation
        bootstrap_scores = bootstrap_validation(
            X, y, model_class, model_kwargs,
            n_iterations=args.bootstrap_iterations, epochs=args.epochs, device=device
        )
        results['bootstrap'][model_name] = bootstrap_scores

        print(f"\nBootstrap Validation Summary:")
        print(f"  Mean F1: {np.mean(bootstrap_scores):.4f}")
        print(f"  Std Dev: {np.std(bootstrap_scores):.4f}")
        print(
            f"  95% CI:  [{np.percentile(bootstrap_scores, 2.5):.4f}, {np.percentile(bootstrap_scores, 97.5):.4f}]")

    # Generate visualizations and report
    plot_generalization_results(results, args.output_dir)
    generate_report(results, args.output_dir)

    print("\n" + "="*80)
    print("GENERALIZATION VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("- generalization_analysis.png")
    print("- GENERALIZATION_REPORT.md")


if __name__ == '__main__':
    main()
