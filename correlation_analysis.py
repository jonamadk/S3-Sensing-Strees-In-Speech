#!/usr/bin/env python3
"""
Pearson Correlation Analysis for Word Stress Features
Generates correlation matrix and heatmap between stress levels and prosodic features.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def load_data(json_path):
    """Load word stress features from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} words from {json_path}")
    return df


def encode_stress_labels(df):
    """
    Encode stress labels for correlation analysis.
    Creates both categorical encoding and separate binary columns.
    Returns tuple: (dataframe, classification_type)
    """
    # Check if we have 2-class or 3-class clustering
    unique_labels = df['stress_label'].unique()

    if 'stressed' in unique_labels and 'unstressed' in unique_labels:
        # Binary classification
        stress_mapping = {
            'unstressed': 0,
            'stressed': 1
        }
        df['stress_numeric'] = df['stress_label'].map(stress_mapping)

        # Binary encoding
        df['is_stressed'] = (df['stress_label'] == 'stressed').astype(int)
        df['is_unstressed'] = (df['stress_label'] == 'unstressed').astype(int)

        return df, 'binary'
    else:
        # 3-class classification
        stress_mapping = {
            'unstressed': 0,
            'secondary': 1,
            'primary': 2
        }
        df['stress_numeric'] = df['stress_label'].map(stress_mapping)

        # Binary encoding for each stress type (for point-biserial correlation)
        df['is_primary'] = (df['stress_label'] == 'primary').astype(int)
        df['is_secondary'] = (df['stress_label'] == 'secondary').astype(int)
        df['is_unstressed'] = (df['stress_label'] == 'unstressed').astype(int)

        return df, '3class'


def calculate_correlations(df, output_dir, classification_type):
    """Calculate Pearson correlations between stress levels and features."""
    print("\n" + "="*80)
    print("PEARSON CORRELATION ANALYSIS")
    print("="*80)

    # Select feature columns (numeric only)
    feature_cols = [
        'word_duration', 'vowel_duration', 'consonant_duration', 'vowel_ratio',
        'num_vowels', 'num_phonemes',
        'pitch_mean', 'pitch_max', 'pitch_min', 'pitch_range', 'pitch_std',
        'pitch_slope', 'pitch_madiff',
        'vowel_pitch_mean', 'vowel_pitch_max',
        'energy_mean', 'energy_max',
        'pre_pause', 'post_pause',
        'pos_norm_start', 'pos_norm_center',
        'prominence_score'
    ]

    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]

    # Calculate correlations with stress_numeric (ordinal)
    correlations_ordinal = {}
    for feature in available_features:
        if df[feature].notna().sum() > 0:
            # Remove rows with NaN or inf values
            valid_mask = df[feature].notna() & np.isfinite(
                df[feature]) & df['stress_numeric'].notna()
            if valid_mask.sum() > 10:  # Need at least 10 valid samples
                feature_data = df.loc[valid_mask, feature]
                stress_data = df.loc[valid_mask, 'stress_numeric']

                # Check for constant values
                if feature_data.std() > 1e-10 and stress_data.std() > 1e-10:
                    corr, pval = pearsonr(feature_data, stress_data)
                    correlations_ordinal[feature] = {
                        'correlation': corr,
                        'p_value': pval,
                        'abs_corr': abs(corr)
                    }

    # Sort by absolute correlation
    sorted_features = sorted(correlations_ordinal.items(),
                             key=lambda x: x[1]['abs_corr'],
                             reverse=True)

    print("\nTop 10 Features Correlated with Stress Level (Ordinal):")
    print(f"{'Feature':<25} {'Correlation':>12} {'P-value':>12}")
    print("-" * 50)
    for feature, stats in sorted_features[:10]:
        sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
        print(
            f"{feature:<25} {stats['correlation']:>12.4f} {stats['p_value']:>12.2e} {sig}")

    # Calculate point-biserial correlations for each stress type
    if classification_type == 'binary':
        stress_types = ['is_stressed', 'is_unstressed']
        stress_labels = ['Stressed', 'Unstressed']
    else:
        stress_types = ['is_primary', 'is_secondary', 'is_unstressed']
        stress_labels = ['Primary', 'Secondary', 'Unstressed']

    correlations_by_type = {}
    for stress_col, label in zip(stress_types, stress_labels):
        correlations_by_type[label] = {}
        for feature in available_features:
            if df[feature].notna().sum() > 0:
                # Remove rows with NaN or inf values
                valid_mask = df[feature].notna() & np.isfinite(
                    df[feature]) & df[stress_col].notna()
                if valid_mask.sum() > 10:
                    feature_data = df.loc[valid_mask, feature]
                    stress_data = df.loc[valid_mask, stress_col]

                    # Check for constant values and variation
                    if feature_data.std() > 1e-10 and stress_data.std() > 1e-10:
                        # Point-biserial correlation (binary vs continuous)
                        corr, pval = pointbiserialr(stress_data, feature_data)
                        correlations_by_type[label][feature] = {
                            'correlation': corr,
                            'p_value': pval
                        }

    # Print top correlations for each stress type
    print("\n" + "="*80)
    for label in stress_labels:
        if label in correlations_by_type and correlations_by_type[label]:
            print(f"\nTop Features for {label} Stress:")
            sorted_corr = sorted(correlations_by_type[label].items(),
                                 key=lambda x: abs(x[1]['correlation']),
                                 reverse=True)
            print(f"{'Feature':<25} {'Correlation':>12} {'P-value':>12}")
            print("-" * 50)
            for feature, stats in sorted_corr[:10]:
                sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                print(
                    f"{feature:<25} {stats['correlation']:>12.4f} {stats['p_value']:>12.2e} {sig}")

    # Save correlation tables
    ordinal_df = pd.DataFrame([
        {'feature': feat,
            'correlation': stats['correlation'], 'p_value': stats['p_value']}
        for feat, stats in correlations_ordinal.items()
    ]).sort_values('correlation', key=abs, ascending=False)

    ordinal_path = os.path.join(output_dir, 'correlations_ordinal.csv')
    ordinal_df.to_csv(ordinal_path, index=False)
    print(f"\nOrdinal correlations saved to: {ordinal_path}")

    # Save correlations by stress type
    for label in stress_labels:
        if label in correlations_by_type and correlations_by_type[label]:
            type_df = pd.DataFrame([
                {'feature': feat,
                    'correlation': stats['correlation'], 'p_value': stats['p_value']}
                for feat, stats in correlations_by_type[label].items()
            ]).sort_values('correlation', key=abs, ascending=False)

            type_path = os.path.join(
                output_dir, f'correlations_{label.lower()}.csv')
            type_df.to_csv(type_path, index=False)

    return correlations_ordinal, correlations_by_type, available_features, stress_labels


def create_correlation_heatmaps(df, correlations_by_type, available_features, stress_labels, output_dir):
    """Create correlation heatmap visualizations."""
    print("\nGenerating correlation heatmaps...")

    # 1. Full correlation matrix heatmap (all features)
    fig, ax = plt.subplots(figsize=(18, 14))

    # Prepare data for heatmap
    correlation_matrix = []
    actual_labels = []

    for label in stress_labels:
        if label in correlations_by_type and correlations_by_type[label]:
            row = [correlations_by_type[label].get(feat, {'correlation': 0})['correlation']
                   for feat in available_features]
            correlation_matrix.append(row)
            actual_labels.append(label)

    # Create DataFrame for heatmap
    corr_df = pd.DataFrame(
        correlation_matrix,
        index=actual_labels,
        columns=[feat.replace('_', ' ').title() for feat in available_features]
    )

    # Plot heatmap
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5, ax=ax)

    ax.set_title('Pearson Correlation: Stress Levels vs Prosodic Features',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Stress Type', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prosodic Features', fontsize=13, fontweight='bold')

    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap_full.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {heatmap_path}")
    plt.close()

    # 2. Top 15 features heatmap (more readable)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Select top 15 features by maximum absolute correlation
    max_corr_per_feature = {}
    for feat in available_features:
        max_corr = 0
        for label in stress_labels:
            if label in correlations_by_type and feat in correlations_by_type[label]:
                corr = abs(correlations_by_type[label][feat]['correlation'])
                if corr > max_corr:
                    max_corr = corr
        max_corr_per_feature[feat] = max_corr

    top_features = sorted(max_corr_per_feature.items(),
                          key=lambda x: x[1], reverse=True)[:15]
    top_feature_names = [feat for feat, _ in top_features]

    # Create correlation matrix for top features
    top_correlation_matrix = []
    top_actual_labels = []
    for label in stress_labels:
        if label in correlations_by_type and correlations_by_type[label]:
            row = [correlations_by_type[label].get(feat, {'correlation': 0})['correlation']
                   for feat in top_feature_names]
            top_correlation_matrix.append(row)
            top_actual_labels.append(label)

    top_corr_df = pd.DataFrame(
        top_correlation_matrix,
        index=top_actual_labels,
        columns=[feat.replace('_', ' ').title() for feat in top_feature_names]
    )

    # Plot heatmap with larger annotations
    sns.heatmap(top_corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'},
                linewidths=1, ax=ax, annot_kws={'size': 10})

    ax.set_title('Top 15 Correlated Features with Stress Levels\n(Pearson Correlation)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Stress Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prosodic Features', fontsize=12, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    top_heatmap_path = os.path.join(
        output_dir, 'correlation_heatmap_top15.png')
    plt.savefig(top_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {top_heatmap_path}")
    plt.close()

    # 3. Comparison bar chart of top correlations
    num_types = len(stress_labels)
    fig, axes = plt.subplots(1, num_types, figsize=(6*num_types, 6))
    if num_types == 1:
        axes = [axes]

    for idx, label in enumerate(stress_labels):
        ax = axes[idx]

        if label in correlations_by_type and correlations_by_type[label]:
            # Get top 10 features for this stress type
            sorted_corr = sorted(correlations_by_type[label].items(),
                                 key=lambda x: abs(x[1]['correlation']),
                                 reverse=True)[:10]

            features = [feat.replace('_', ' ').title()
                        for feat, _ in sorted_corr]
            correlations = [stats['correlation'] for _, stats in sorted_corr]

            # Color bars by sign
            colors = ['#d62728' if c < 0 else '#2ca02c' for c in correlations]

            # Horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, correlations, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Correlation', fontsize=10)
            ax.set_title(f'{label} Stress\nTop 10 Features',
                         fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, v in enumerate(correlations):
                ax.text(v + 0.02 if v >= 0 else v - 0.02, i, f'{v:.3f}',
                        va='center', ha='left' if v >= 0 else 'right',
                        fontsize=8, fontweight='bold')

    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'correlation_bar_charts.png')
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {bar_chart_path}")
    plt.close()

    # 4. Feature correlation comparison across stress types
    if num_types > 1:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Select top features and prepare data
        x = np.arange(len(top_feature_names))
        width = 0.8 / num_types

        colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd']

        for idx, label in enumerate(stress_labels):
            if label in correlations_by_type:
                corr_values = [correlations_by_type[label].get(feat, {'correlation': 0})['correlation']
                               for feat in top_feature_names]
                ax.bar(x + (idx - num_types/2 + 0.5) * width, corr_values, width,
                       label=label, alpha=0.8, color=colors[idx % len(colors)])

        ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prosodic Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Correlations Across Stress Types\n(Top 15 Features)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([feat.replace('_', ' ').title() for feat in top_feature_names],
                           rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        grouped_bar_path = os.path.join(
            output_dir, 'correlation_grouped_bars.png')
        plt.savefig(grouped_bar_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {grouped_bar_path}")
        plt.close()


def generate_correlation_report(correlations_ordinal, correlations_by_type, stress_labels, output_dir):
    """Generate markdown report with correlation analysis."""
    print("\nGenerating correlation analysis report...")

    report_lines = []
    report_lines.append("# Pearson Correlation Analysis Report")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(
        "This report analyzes Pearson correlations between word stress levels")
    report_lines.append(
        "and prosodic features extracted from the word_stress_features dataset.")
    report_lines.append("")

    report_lines.append("## Methodology")
    report_lines.append("")
    report_lines.append(
        "- **Ordinal Correlation**: Treats stress as ordinal")
    report_lines.append(
        "- **Point-Biserial Correlation**: Binary correlation for each stress type separately")
    report_lines.append(
        "- **Significance Level**: *** p<0.001, ** p<0.01, * p<0.05")
    report_lines.append("")

    # Top features overall
    if correlations_ordinal:
        sorted_features = sorted(correlations_ordinal.items(),
                                 key=lambda x: abs(x[1]['correlation']),
                                 reverse=True)

        report_lines.append("## Top 10 Most Correlated Features (Ordinal)")
        report_lines.append("")
        report_lines.append(
            "| Rank | Feature | Correlation | P-value | Significance |")
        report_lines.append(
            "|------|---------|-------------|---------|--------------|")

        for rank, (feature, stats) in enumerate(sorted_features[:10], 1):
            sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "-"
            report_lines.append(
                f"| {rank} | {feature} | {stats['correlation']:.4f} | {stats['p_value']:.2e} | {sig} |")

        report_lines.append("")

    # Top features for each stress type
    for label in stress_labels:
        if label in correlations_by_type and correlations_by_type[label]:
            sorted_corr = sorted(correlations_by_type[label].items(),
                                 key=lambda x: abs(x[1]['correlation']),
                                 reverse=True)

            report_lines.append(f"## Top 10 Features for {label} Stress")
            report_lines.append("")
            report_lines.append(
                "| Rank | Feature | Correlation | P-value | Significance |")
            report_lines.append(
                "|------|---------|-------------|---------|--------------|")

            for rank, (feature, stats) in enumerate(sorted_corr[:10], 1):
                sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                report_lines.append(
                    f"| {rank} | {feature} | {stats['correlation']:.4f} | {stats['p_value']:.2e} | {sig} |")

            report_lines.append("")

    report_lines.append("## Key Findings")
    report_lines.append("")

    if correlations_ordinal:
        # Identify strongest positive and negative correlations
        max_pos = max(correlations_ordinal.items(),
                      key=lambda x: x[1]['correlation'])
        max_neg = min(correlations_ordinal.items(),
                      key=lambda x: x[1]['correlation'])

        report_lines.append(f"### Strongest Positive Correlation")
        report_lines.append(
            f"- **{max_pos[0]}**: r = {max_pos[1]['correlation']:.4f} (p = {max_pos[1]['p_value']:.2e})")
        report_lines.append(
            f"  - Higher values associated with higher stress levels")
        report_lines.append("")

        report_lines.append(f"### Strongest Negative Correlation")
        report_lines.append(
            f"- **{max_neg[0]}**: r = {max_neg[1]['correlation']:.4f} (p = {max_neg[1]['p_value']:.2e})")
        report_lines.append(
            f"  - Higher values associated with lower stress levels")
        report_lines.append("")

    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append("**Features positively correlated with stress:**")
    report_lines.append("- Indicate characteristics of stressed syllables")
    report_lines.append(
        "- Typically: higher pitch, longer duration, higher prominence")
    report_lines.append("")
    report_lines.append("**Features negatively correlated with stress:**")
    report_lines.append("- Indicate characteristics of unstressed syllables")
    report_lines.append(
        "- Typically: position in utterance, certain pause patterns")
    report_lines.append("")

    # Save report
    report_path = os.path.join(output_dir, 'CORRELATION_ANALYSIS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Correlation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Pearson correlation analysis for word stress features')
    parser.add_argument('--input-json', required=True,
                        help='Path to word_stress_features.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for visualizations and reports')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("PEARSON CORRELATION ANALYSIS FOR WORD STRESS")
    print("="*80)

    # Load data
    df = load_data(args.input_json)

    # Encode stress labels
    df, classification_type = encode_stress_labels(df)

    # Calculate correlations
    correlations_ordinal, correlations_by_type, available_features, stress_labels = \
        calculate_correlations(df, args.output_dir, classification_type)

    # Create visualizations
    create_correlation_heatmaps(
        df, correlations_by_type, available_features, stress_labels, args.output_dir)

    # Generate report
    generate_correlation_report(
        correlations_ordinal, correlations_by_type, stress_labels, args.output_dir)

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {args.output_dir}:")
    print("  - correlation_heatmap_full.png")
    print("  - correlation_heatmap_top15.png")
    print("  - correlation_bar_charts.png")
    if classification_type == '3class':
        print("  - correlation_grouped_bars.png")
    print("  - correlations_ordinal.csv")
    for label in stress_labels:
        print(f"  - correlations_{label.lower()}.csv")
    print("  - CORRELATION_ANALYSIS_REPORT.md")


if __name__ == '__main__':
    main()
