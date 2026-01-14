#!/usr/bin/env python3
"""
Machine Learning Models for Word Stress Prediction
Trains and evaluates multiple ML models to predict word stress from prosodic features.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_data(json_path):
    """Load word stress features from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} words from {json_path}")
    return df


def prepare_features(df):
    """Prepare features and labels for ML models."""
    # Feature columns
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

    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]

    # Extract features
    X = df[available_features].copy()

    # Handle NaN and inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Get labels
    y = df['stress_label'].copy()

    # Detect classification type
    unique_labels = y.unique()
    if 'stressed' in unique_labels and 'unstressed' in unique_labels:
        classification_type = 'binary'
        # Map to numeric
        label_mapping = {'unstressed': 0, 'stressed': 1}
        class_names = ['Unstressed', 'Stressed']
    else:
        classification_type = '3class'
        # Map to numeric
        label_mapping = {'unstressed': 0, 'secondary': 1, 'primary': 2}
        class_names = ['Unstressed', 'Secondary', 'Primary']

    y = y.map(label_mapping)

    print(f"\nClassification type: {classification_type}")
    print(f"Features: {len(available_features)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    return X, y, available_features, class_names, classification_type


def train_models(X_train, X_test, y_train, y_test, classification_type):
    """Train all ML models and return results."""
    print("\n" + "="*80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*80)

    models = {}
    results = {}
    training_history = {}

    # 1. K-Nearest Neighbors
    print("\n[1/6] Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(
        n_neighbors=5, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    y_pred = knn.predict(X_test)
    results['KNN'] = evaluate_model(y_test, y_pred, 'KNN', classification_type)

    # 2. Decision Tree
    print("\n[2/6] Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    y_pred = dt.predict(X_test)
    results['Decision Tree'] = evaluate_model(
        y_test, y_pred, 'Decision Tree', classification_type)

    # 3. Random Forest
    print("\n[3/6] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    y_pred = rf.predict(X_test)
    results['Random Forest'] = evaluate_model(
        y_test, y_pred, 'Random Forest', classification_type)

    # 4. Naive Bayes
    print("\n[4/6] Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    y_pred = nb.predict(X_test)
    results['Naive Bayes'] = evaluate_model(
        y_test, y_pred, 'Naive Bayes', classification_type)

    # 5. Neural Network
    print("\n[5/6] Training Neural Network...")
    nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                       max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
    nn.fit(X_train, y_train)
    models['Neural Network'] = nn
    y_pred = nn.predict(X_test)
    results['Neural Network'] = evaluate_model(
        y_test, y_pred, 'Neural Network', classification_type)
    # Store loss curve
    if hasattr(nn, 'loss_curve_'):
        training_history['Neural Network'] = {'loss_curve': nn.loss_curve_}

    # 6. XGBoost
    print("\n[6/6] Training XGBoost...")
    if classification_type == 'binary':
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                      random_state=42, eval_metric='logloss')
    else:
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                      random_state=42, eval_metric='mlogloss', num_class=3)

    # Train with validation set to track performance
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    models['XGBoost'] = xgb_model
    y_pred = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model(
        y_test, y_pred, 'XGBoost', classification_type)
    # Store evaluation results
    if hasattr(xgb_model, 'evals_result_'):
        training_history['XGBoost'] = xgb_model.evals_result_

    return models, results, training_history


def evaluate_model(y_true, y_pred, model_name, classification_type):
    """Evaluate model performance."""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    if classification_type == 'binary':
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


def create_visualizations(results, class_names, output_dir):
    """Create comprehensive visualizations."""
    print("\nGenerating visualizations...")

    # 1. Model Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [results[model][metric] for model in model_names]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))

        bars = ax.bar(model_names, values, color=colors, alpha=0.8)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {comparison_path}")
    plt.close()

    # 2. Confusion Matrices
    num_models = len(results)
    cols = 3
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
    axes = axes.flatten() if num_models > 1 else [axes]

    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        cm = result['confusion_matrix']

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Percentage'}, ax=ax)

        ax.set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {cm_path}")
    plt.close()

    # 3. Overall Performance Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_data = []
    for model_name in model_names:
        metrics_data.append([
            results[model_name]['accuracy'],
            results[model_name]['precision'],
            results[model_name]['recall'],
            results[model_name]['f1']
        ])

    metrics_df = pd.DataFrame(metrics_data,
                              index=model_names,
                              columns=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=1, ax=ax,
                cbar_kws={'label': 'Score'})

    ax.set_title('Model Performance Heatmap',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Model', fontsize=11, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {heatmap_path}")
    plt.close()


def plot_learning_curves(models, X_train, y_train, output_dir):
    """Plot learning curves to detect overfitting/underfitting."""
    print("\nGenerating learning curves...")

    num_models = len(models)
    cols = 2
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6*rows))
    axes = axes.flatten() if num_models > 1 else [axes]

    train_sizes = np.linspace(0.1, 1.0, 10)

    for idx, (model_name, model) in enumerate(models.items()):
        ax = axes[idx]

        print(f"  - Computing learning curve for {model_name}...")

        # Compute learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42
        )

        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#2ca02c',
                label='Training score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='#2ca02c')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='#d62728',
                label='Validation score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='#d62728')

        # Detect overfitting
        gap = train_mean[-1] - val_mean[-1]
        if gap > 0.1:
            status = "⚠️ Overfitting"
            color = 'red'
        elif gap > 0.05:
            status = "⚡ Slight Overfitting"
            color = 'orange'
        else:
            status = "✓ Good Fit"
            color = 'green'

        ax.set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\n{status} (Gap: {gap:.3f})',
                     fontsize=12, fontweight='bold', color=color)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    learning_curve_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {learning_curve_path}")
    plt.close()


def plot_training_history(training_history, output_dir):
    """Plot training history for models that support it."""
    if not training_history:
        return

    print("\nGenerating training history plots...")

    # Neural Network Loss Curve
    if 'Neural Network' in training_history:
        fig, ax = plt.subplots(figsize=(10, 6))

        loss_curve = training_history['Neural Network']['loss_curve']
        epochs = range(1, len(loss_curve) + 1)

        ax.plot(epochs, loss_curve, 'o-', color='#1f77b4',
                linewidth=2, markersize=4, label='Training Loss')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Neural Network: Training Loss Over Epochs',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add convergence annotation
        final_loss = loss_curve[-1]
        ax.axhline(y=final_loss, color='red', linestyle='--', alpha=0.5,
                   label=f'Final Loss: {final_loss:.4f}')
        ax.legend(fontsize=11)

        plt.tight_layout()
        nn_path = os.path.join(output_dir, 'neural_network_loss.png')
        plt.savefig(nn_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {nn_path}")
        plt.close()

    # XGBoost Training vs Validation
    if 'XGBoost' in training_history:
        fig, ax = plt.subplots(figsize=(10, 6))

        evals = training_history['XGBoost']

        # Get the metric name (first key in the dict)
        train_key = list(evals.keys())[0]
        val_key = list(evals.keys())[1]
        metric_name = list(evals[train_key].keys())[0]

        train_metric = evals[train_key][metric_name]
        val_metric = evals[val_key][metric_name]
        epochs = range(1, len(train_metric) + 1)

        ax.plot(epochs, train_metric, 'o-', color='#2ca02c',
                linewidth=2, markersize=3, label='Training', alpha=0.8)
        ax.plot(epochs, val_metric, 'o-', color='#d62728',
                linewidth=2, markersize=3, label='Validation', alpha=0.8)

        # Highlight best iteration
        best_iter = np.argmin(
            val_metric) if 'loss' in metric_name else np.argmax(val_metric)
        ax.axvline(x=best_iter+1, color='blue', linestyle='--', alpha=0.5,
                   label=f'Best Iteration: {best_iter+1}')

        ax.set_xlabel('Boosting Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name.upper(), fontsize=12, fontweight='bold')
        ax.set_title('XGBoost: Training vs Validation Performance',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add overfitting indicator
        gap = abs(train_metric[-1] - val_metric[-1])
        gap_pct = (gap / abs(train_metric[-1])) * \
            100 if train_metric[-1] != 0 else 0

        if gap_pct > 10:
            status_text = f"⚠️ Overfitting Detected (Gap: {gap_pct:.1f}%)"
            status_color = 'red'
        elif gap_pct > 5:
            status_text = f"⚡ Slight Overfitting (Gap: {gap_pct:.1f}%)"
            status_color = 'orange'
        else:
            status_text = f"✓ Good Fit (Gap: {gap_pct:.1f}%)"
            status_color = 'green'

        ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=status_color,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        xgb_path = os.path.join(output_dir, 'xgboost_training.png')
        plt.savefig(xgb_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {xgb_path}")
        plt.close()


def save_results(results, models, scaler, feature_names, class_names, output_dir):
    """Save models and results."""
    print("\nSaving models and results...")

    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = models[best_model_name]

    # Save best model
    model_path = os.path.join(output_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"  - Best model ({best_model_name}) saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  - Scaler saved to: {scaler_path}")

    # Save all models
    for model_name, model in models.items():
        safe_name = model_name.replace(' ', '_').lower()
        path = os.path.join(output_dir, f'model_{safe_name}.pkl')
        joblib.dump(model, path)

    # Save results CSV
    results_data = []
    for model_name, result in results.items():
        results_data.append({
            'model': model_name,
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1']
        })

    results_df = pd.DataFrame(results_data).sort_values(
        'f1_score', ascending=False)
    results_path = os.path.join(output_dir, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  - Results saved to: {results_path}")

    # Save model info
    info = {
        'best_model': best_model_name,
        'feature_names': feature_names,
        'class_names': class_names,
        'num_features': len(feature_names),
        'num_classes': len(class_names)
    }

    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  - Model info saved to: {info_path}")

    return best_model_name, results_df


def generate_report(results, results_df, class_names, best_model_name, output_dir):
    """Generate markdown report."""
    print("\nGenerating analysis report...")

    report_lines = []
    report_lines.append("# Machine Learning Models for Word Stress Prediction")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(
        "This report presents the performance of six machine learning models")
    report_lines.append(
        "trained to predict word stress from prosodic features.")
    report_lines.append("")

    report_lines.append("## Models Evaluated")
    report_lines.append("")
    report_lines.append(
        "1. **K-Nearest Neighbors (KNN)** - Instance-based learning")
    report_lines.append("2. **Decision Tree** - Rule-based classification")
    report_lines.append("3. **Random Forest** - Ensemble of decision trees")
    report_lines.append("4. **Naive Bayes** - Probabilistic classifier")
    report_lines.append(
        "5. **Neural Network (MLP)** - Multi-layer perceptron with 3 hidden layers")
    report_lines.append("6. **XGBoost** - Gradient boosting ensemble")
    report_lines.append("")

    report_lines.append("## Performance Summary")
    report_lines.append("")
    report_lines.append(
        "| Rank | Model | Accuracy | Precision | Recall | F1 Score |")
    report_lines.append(
        "|------|-------|----------|-----------|--------|----------|")

    for idx, row in results_df.iterrows():
        rank = idx + 1
        report_lines.append(
            f"| {rank} | {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1_score']:.4f} |"
        )

    report_lines.append("")
    report_lines.append(f"## Best Model: {best_model_name}")
    report_lines.append("")

    best_result = results[best_model_name]
    report_lines.append(f"- **Accuracy**: {best_result['accuracy']:.4f}")
    report_lines.append(f"- **Precision**: {best_result['precision']:.4f}")
    report_lines.append(f"- **Recall**: {best_result['recall']:.4f}")
    report_lines.append(f"- **F1 Score**: {best_result['f1']:.4f}")
    report_lines.append("")

    report_lines.append("## Detailed Results by Model")
    report_lines.append("")

    for model_name, result in results.items():
        report_lines.append(f"### {model_name}")
        report_lines.append("")
        report_lines.append("**Metrics:**")
        report_lines.append(f"- Accuracy: {result['accuracy']:.4f}")
        report_lines.append(f"- Precision: {result['precision']:.4f}")
        report_lines.append(f"- Recall: {result['recall']:.4f}")
        report_lines.append(f"- F1 Score: {result['f1']:.4f}")
        report_lines.append("")

        report_lines.append("**Confusion Matrix:**")
        report_lines.append("```")
        cm = result['confusion_matrix']
        report_lines.append(
            f"Predicted:  {' '.join([f'{cn:>12}' for cn in class_names])}")
        for idx, row in enumerate(cm):
            report_lines.append(
                f"Actual {class_names[idx]:>10}: {' '.join([f'{val:>12}' for val in row])}")
        report_lines.append("```")
        report_lines.append("")

    report_lines.append("## Key Findings")
    report_lines.append("")
    report_lines.append("### Model Strengths")
    report_lines.append("")

    # Identify best model for each metric
    best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_prec = max(results.items(), key=lambda x: x[1]['precision'])[0]
    best_rec = max(results.items(), key=lambda x: x[1]['recall'])[0]
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])[0]

    report_lines.append(
        f"- **Highest Accuracy**: {best_acc} ({results[best_acc]['accuracy']:.4f})")
    report_lines.append(
        f"- **Highest Precision**: {best_prec} ({results[best_prec]['precision']:.4f})")
    report_lines.append(
        f"- **Highest Recall**: {best_rec} ({results[best_rec]['recall']:.4f})")
    report_lines.append(
        f"- **Highest F1 Score**: {best_f1} ({results[best_f1]['f1']:.4f})")
    report_lines.append("")

    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append(
        f"1. **Deploy {best_model_name}** for production use (best overall performance)")
    report_lines.append("2. Consider ensemble methods for improved robustness")
    report_lines.append("3. Use cross-validation for more reliable estimates")
    report_lines.append(
        "4. Feature engineering could further improve performance")
    report_lines.append("")

    report_lines.append("## Usage")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("import joblib")
    report_lines.append("import pandas as pd")
    report_lines.append("")
    report_lines.append("# Load model and scaler")
    report_lines.append("model = joblib.load('best_model.pkl')")
    report_lines.append("scaler = joblib.load('scaler.pkl')")
    report_lines.append("")
    report_lines.append("# Prepare features")
    report_lines.append("features = pd.DataFrame([{...}])  # Your features")
    report_lines.append("features_scaled = scaler.transform(features)")
    report_lines.append("")
    report_lines.append("# Predict")
    report_lines.append("prediction = model.predict(features_scaled)")
    report_lines.append("probability = model.predict_proba(features_scaled)")
    report_lines.append("```")
    report_lines.append("")

    # Save report
    report_path = os.path.join(output_dir, 'ML_MODELS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ML models for word stress prediction')
    parser.add_argument('--input-json', required=True,
                        help='Path to word_stress_features.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for models and results')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("MACHINE LEARNING MODELS FOR WORD STRESS PREDICTION")
    print("="*80)

    # Load and prepare data
    df = load_data(args.input_json)
    X, y, feature_names, class_names, classification_type = prepare_features(
        df)

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

    # Train models
    models, results, training_history = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test, classification_type)

    # Create visualizations
    create_visualizations(results, class_names, args.output_dir)

    # Plot learning curves (overfitting/underfitting analysis)
    plot_learning_curves(models, X_train_scaled, y_train, args.output_dir)

    # Plot training history for Neural Network and XGBoost
    plot_training_history(training_history, args.output_dir)

    # Save models and results
    best_model_name, results_df = save_results(results, models, scaler, feature_names,
                                               class_names, args.output_dir)

    # Generate report
    generate_report(results, results_df, class_names,
                    best_model_name, args.output_dir)

    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
