#!/usr/bin/env python3
"""
Word Stress Clustering based on Vowel-Level Prosodic Features
Uses vowel duration, pitch dynamics, and prominence scoring for
accurate stress detection (primary, secondary, unstressed).
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from collections import defaultdict


# --------------------------------------------------------------
# Vowel Detection
# --------------------------------------------------------------
VOWELS = set("""
AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW
AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 EH0 EH1 EH2 ER0 ER1 ER2
EY0 EY1 EY2 IH0 IH1 IH2 IY0 IY1 IY2 OW0 OW1 OW2 OY0 OY1 OY2 UH0 UH1 UH2 UW0 UW1 UW2
""".split())
# Add IPA-like symbols
VOWELS.update({"o", "ᵻ", "a", "e", "ɐ", "ɒ", "ɜ"})


def is_vowel(ph):
    """Check if phoneme is a vowel."""
    base = ''.join([c for c in ph if not c.isdigit()])
    return base in VOWELS


def safe_pitch(p):
    """Return valid pitch or 0."""
    if p is None or p == 0:
        return 0.0
    return float(p)


def load_phoneme_data(json_path: str) -> list:
    """Load phoneme features from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def aggregate_word_features(phonemes: list) -> pd.DataFrame:
    """
    Aggregate phoneme-level features to word-level with vowel-based analysis.
    Computes duration, pitch dynamics, and prominence features.
    """
    if not phonemes:
        return pd.DataFrame()

    # Utterance-level context
    utter_start = min(p['start'] for p in phonemes)
    utter_end = max(p['end'] for p in phonemes)
    utter_dur = max(1e-6, utter_end - utter_start)

    # Group by consecutive word instances
    words_data = []
    current_word = None
    current_data = None

    for phon in phonemes:
        word = phon['word']

        # New word instance
        if word != current_word or current_data is None:
            if current_data is not None:
                words_data.append(current_data)

            current_word = word
            current_data = {
                'word': word,
                'phonemes': [],
                'starts': [],
                'ends': [],
                'pitches': [],
                'energies': [],
                'sentence': phon.get('sentence', '')
            }

        # Add phoneme data
        current_data['phonemes'].append(phon['char'])
        current_data['starts'].append(phon['start'])
        current_data['ends'].append(phon['end'])
        current_data['pitches'].append(safe_pitch(phon.get('pitch', 0)))
        current_data['energies'].append(phon.get('energy', 0))

    # Don't forget last word
    if current_data is not None:
        words_data.append(current_data)

    # Sort by start time
    words_data.sort(key=lambda w: min(w['starts']))

    # Extract features for each word
    features = []
    for i, data in enumerate(words_data):
        phones = np.array(data['phonemes'])
        starts = np.array(data['starts'], dtype=float)
        ends = np.array(data['ends'], dtype=float)
        durs = ends - starts
        pitch = np.array(data['pitches'], dtype=float)
        energy = np.array(data['energies'], dtype=float)

        # Word boundaries
        w_start = float(starts.min())
        w_end = float(ends.max())
        word_dur = max(1e-6, w_end - w_start)

        # Vowel/consonant separation
        vowel_mask = np.array([is_vowel(ph) for ph in phones], dtype=bool)
        cons_mask = ~vowel_mask

        # Duration features
        vowel_dur = float(durs[vowel_mask].sum()) if vowel_mask.any() else 0.0
        cons_dur = float(durs[cons_mask].sum()) if cons_mask.any() else 0.0
        vowel_ratio = vowel_dur / word_dur
        num_vowels = int(vowel_mask.sum())
        num_phonemes = len(phones)

        # Pitch features (all phonemes)
        valid_pitch = pitch[pitch > 0]
        pitch_mean = float(valid_pitch.mean()) if len(valid_pitch) else 0.0
        pitch_max = float(valid_pitch.max()) if len(valid_pitch) else 0.0
        pitch_min = float(valid_pitch.min()) if len(valid_pitch) else 0.0
        pitch_range = pitch_max - pitch_min
        pitch_std = float(valid_pitch.std()) if len(valid_pitch) > 1 else 0.0
        pitch_slope = (pitch[-1] - pitch[0]) / word_dur if len(
            pitch) > 1 and pitch[0] > 0 and pitch[-1] > 0 else 0.0
        pitch_madiff = float(np.mean(np.abs(np.diff(valid_pitch)))) if len(
            valid_pitch) > 1 else 0.0

        # Vowel-only pitch
        if vowel_mask.any():
            v_pitch = pitch[vowel_mask]
            v_pitch = v_pitch[v_pitch > 0]
            vowel_pitch_mean = float(v_pitch.mean()) if len(
                v_pitch) else pitch_mean
            vowel_pitch_max = float(v_pitch.max()) if len(
                v_pitch) else pitch_max
        else:
            vowel_pitch_mean = pitch_mean
            vowel_pitch_max = pitch_max

        # Energy features
        energy_mean = float(energy.mean()) if len(energy) else 0.0
        energy_max = float(energy.max()) if len(energy) else 0.0

        # Pause features
        pre_pause = 0.0
        if i > 0:
            prev_end = max(words_data[i-1]['ends'])
            pre_pause = max(0.0, w_start - prev_end)

        post_pause = 0.0
        if i < len(words_data) - 1:
            next_start = min(words_data[i+1]['starts'])
            post_pause = max(0.0, next_start - w_end)

        # Position features
        pos_norm_start = (w_start - utter_start) / utter_dur
        pos_norm_center = ((w_start + w_end) / 2 - utter_start) / utter_dur

        features.append({
            'word': data['word'],
            'word_start': w_start,
            'word_end': w_end,
            'word_duration': word_dur,
            'sentence': data['sentence'],
            'num_phonemes': num_phonemes,
            'num_vowels': num_vowels,
            'phonemes': ''.join(phones),
            'vowel_duration': vowel_dur,
            'consonant_duration': cons_dur,
            'vowel_ratio': vowel_ratio,
            'pitch_mean': pitch_mean,
            'pitch_max': pitch_max,
            'pitch_min': pitch_min,
            'pitch_range': pitch_range,
            'pitch_slope': pitch_slope,
            'pitch_std': pitch_std,
            'pitch_madiff': pitch_madiff,
            'vowel_pitch_mean': vowel_pitch_mean,
            'vowel_pitch_max': vowel_pitch_max,
            'energy_mean': energy_mean,
            'energy_max': energy_max,
            'pre_pause': pre_pause,
            'post_pause': post_pause,
            'pos_norm_start': pos_norm_start,
            'pos_norm_center': pos_norm_center
        })

    df = pd.DataFrame(features)

    # Z-score normalization for prominence calculation
    for col in ['word_duration', 'vowel_duration', 'vowel_ratio', 'pitch_mean',
                'pitch_max', 'pitch_range', 'pitch_slope', 'pitch_std',
                'pitch_madiff', 'vowel_pitch_mean', 'vowel_pitch_max',
                'pre_pause', 'energy_max']:
        mu = df[col].mean()
        sd = df[col].std() if df[col].std() > 1e-9 else 1.0
        df[f'{col}_z'] = (df[col] - mu) / sd

    # Prominence score: duration + pitch + pause
    df['prominence_score'] = (
        0.5 * df['vowel_duration_z'] +
        0.5 * df['pitch_max_z'] +
        0.3 * df['pitch_range_z'] +
        0.2 * df['pitch_slope_z'] +
        0.3 * df['pre_pause_z'] +
        0.2 * df['energy_max_z']
    )

    return df.sort_values('word_start').reset_index(drop=True)


def cluster_stress_levels(df: pd.DataFrame, n_clusters: int = 2, method: str = 'kmeans') -> tuple:
    """
    Cluster words by prominence score using KMeans or Hierarchical clustering.

    Args:
        df: DataFrame with vowel-based features and prominence_score
        n_clusters: Number of clusters (2 for stressed/unstressed, 3 for primary/secondary/unstressed)
        method: Clustering method ('kmeans' or 'hierarchical')

    Returns:
        Tuple of (DataFrame with cluster labels, clustering_metrics dict, X_scaled, scaler)
    """
    if df.empty:
        return df, {}, None, None

    # Select features for clustering
    feat_cols = ['vowel_duration', 'vowel_ratio', 'pitch_mean', 'pitch_max',
                 'pitch_range', 'pitch_slope', 'pitch_std', 'pre_pause', 'energy_max']

    # Ensure columns exist
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering based on method
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        df['cluster'] = clusterer.fit_predict(X_scaled)
        cluster_col = 'cluster'
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')
        df['hierarchical_cluster'] = clusterer.fit_predict(X_scaled)
        cluster_col = 'hierarchical_cluster'
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'kmeans' or 'hierarchical'")

    # Map clusters to stress labels based on prominence score
    cluster_prominence = df.groupby(
        cluster_col)['prominence_score'].mean().sort_values()

    if n_clusters == 2:
        # stressed/unstressed
        mapping = {
            cluster_prominence.index[0]: 'unstressed',
            cluster_prominence.index[1]: 'stressed'
        }
    elif n_clusters == 3:
        # primary/secondary/unstressed
        mapping = {
            cluster_prominence.index[0]: 'unstressed',
            cluster_prominence.index[1]: 'secondary',
            cluster_prominence.index[2]: 'primary'
        }
    else:
        # Generic labels
        mapping = {c: f'level_{i}' for i,
                   c in enumerate(cluster_prominence.index)}

    label_col = f'{method}_stress_label'
    df[label_col] = df[cluster_col].map(mapping)

    # Calculate clustering metrics
    metrics = {
        'method': method,
        'n_clusters': n_clusters,
        'silhouette': silhouette_score(X_scaled, df[cluster_col]),
        'davies_bouldin': davies_bouldin_score(X_scaled, df[cluster_col]),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, df[cluster_col])
    }

    return df, metrics, X_scaled, scaler


def visualize_stress_clusters(df: pd.DataFrame, output_dir: str):
    """Create visualizations of vowel-based stress clustering results."""

    # Filter valid data
    df_valid = df[df['cluster'] >= 0].copy()

    if df_valid.empty:
        print("No valid data to visualize")
        return

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Scatter: Vowel duration vs Pitch max colored by stress
    ax1 = axes[0, 0]
    for stress in df_valid['stress_label'].unique():
        mask = df_valid['stress_label'] == stress
        ax1.scatter(
            df_valid[mask]['vowel_duration'],
            df_valid[mask]['pitch_max'],
            label=stress.capitalize(),
            alpha=0.6,
            s=100
        )
    ax1.set_xlabel('Vowel Duration (s)', fontsize=12)
    ax1.set_ylabel('Max Pitch (Hz)', fontsize=12)
    ax1.set_title('Word Stress Clustering: Vowel Duration vs Pitch',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot: Prominence score by stress
    ax2 = axes[0, 1]
    df_valid.boxplot(column='prominence_score', by='stress_label', ax=ax2)
    ax2.set_xlabel('Stress Level', fontsize=12)
    ax2.set_ylabel('Prominence Score', fontsize=12)
    ax2.set_title('Prominence Score by Stress Level',
                  fontsize=14, fontweight='bold')
    plt.sca(ax2)
    plt.xticks(rotation=0)

    # 3. Box plot: Vowel ratio by stress
    ax3 = axes[1, 0]
    df_valid.boxplot(column='vowel_ratio', by='stress_label', ax=ax3)
    ax3.set_xlabel('Stress Level', fontsize=12)
    ax3.set_ylabel('Vowel Ratio', fontsize=12)
    ax3.set_title('Vowel Ratio by Stress Level',
                  fontsize=14, fontweight='bold')
    plt.sca(ax3)
    plt.xticks(rotation=0)

    # 4. Bar plot: Stress distribution
    ax4 = axes[1, 1]
    stress_counts = df_valid['stress_label'].value_counts()
    colors = {'primary': '#ff7f0e', 'secondary': '#2ca02c',
              'unstressed': '#1f77b4', 'stressed': '#d62728'}
    stress_counts.plot(kind='bar', ax=ax4,
                       color=[colors.get(x, 'gray') for x in stress_counts.index])
    ax4.set_xlabel('Stress Level', fontsize=12)
    ax4.set_ylabel('Number of Words', fontsize=12)
    ax4.set_title('Stress Level Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

    # Add count labels on bars
    for i, v in enumerate(stress_counts.values):
        ax4.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'word_stress_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    # Create additional detailed visualization
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Scatter with bubble size for vowel ratio
    for stress in df_valid['stress_label'].unique():
        mask = df_valid['stress_label'] == stress
        ax.scatter(
            df_valid[mask]['pitch_max'],
            df_valid[mask]['prominence_score'],
            s=df_valid[mask]['vowel_ratio'] * 1000,  # Size by vowel ratio
            label=stress.capitalize(),
            alpha=0.5
        )

    ax.set_xlabel('Max Pitch (Hz)', fontsize=13)
    ax.set_ylabel('Prominence Score', fontsize=13)
    ax.set_title('Word Stress: Pitch, Prominence, and Vowel Ratio\n(Bubble size = vowel ratio)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path2 = os.path.join(output_dir, 'word_stress_detailed.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to: {output_path2}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Cluster word stress from phoneme features using vowel-based analysis')
    parser.add_argument('--input-json', required=True,
                        help='Input phoneme features JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of stress levels (2 for stressed/unstressed, 3 for primary/secondary/unstressed)')
    parser.add_argument('--method', choices=['kmeans', 'hierarchical', 'both'], default='both',
                        help='Clustering method to use (default: both)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading phoneme data from: {args.input_json}")
    phonemes = load_phoneme_data(args.input_json)

    print(f"Extracting vowel-based features from {len(phonemes)} phonemes...")
    word_df = aggregate_word_features(phonemes)

    # Store all metrics for comparison
    all_metrics = []

    # Run clustering based on method
    methods_to_run = [
        'kmeans', 'hierarchical'] if args.method == 'both' else [args.method]

    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(
            f"Clustering {len(word_df)} words into {args.n_clusters} stress levels using {method.upper()}...")
        print('='*70)

        word_df, metrics, X_scaled, scaler = cluster_stress_levels(
            word_df, args.n_clusters, method=method)
        all_metrics.append(metrics)

        # Print metrics
        print(f"\nClustering Quality Metrics ({method.upper()}):")
        print(
            f"  Silhouette Score: {metrics['silhouette']:.4f} (higher is better, range: -1 to 1)")
        print(
            f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)")
        print(
            f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f} (higher is better)")

        # Print distribution for this method
        label_col = f'{method}_stress_label'
        print(f"\nStress distribution ({method}):")
        print(word_df[label_col].value_counts())

        # Average features by stress level
        print(f"\nAverage features by stress level ({method}):")
        summary_cols = ['vowel_duration', 'vowel_ratio', 'pitch_max',
                        'pitch_range', 'prominence_score']
        available_cols = [c for c in summary_cols if c in word_df.columns]
        if available_cols:
            summary = word_df.groupby(
                label_col)[available_cols].mean().round(3)
            print(summary)

        # Sample words from each category
        print(f"\nSample words by stress level ({method}):")
        for stress in word_df[label_col].unique():
            samples = word_df[word_df[label_col] ==
                              stress]['word'].head(10).tolist()
            print(f"  {stress.upper()}: {', '.join(samples)}")

    # Save comparison metrics
    if len(all_metrics) > 1:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(
            args.output_dir, 'clustering_comparison.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n{'='*70}")
        print("CLUSTERING METHOD COMPARISON")
        print('='*70)
        print(metrics_df.to_string(index=False))
        print(f"\nComparison metrics saved to: {metrics_path}")

    # Prepare final dataset with primary method (for backward compatibility)
    if 'kmeans_stress_label' in word_df.columns:
        word_df['stress_label'] = word_df['kmeans_stress_label']
        word_df['cluster'] = word_df['cluster']
    elif 'hierarchical_stress_label' in word_df.columns:
        word_df['stress_label'] = word_df['hierarchical_stress_label']
        word_df['cluster'] = word_df['hierarchical_cluster']

    # Save word-level dataset
    csv_path = os.path.join(args.output_dir, 'word_stress_features.csv')
    word_df.to_csv(csv_path, index=False)
    print(f"\nWord-level features saved to: {csv_path}")

    # Save JSON format
    json_path = os.path.join(args.output_dir, 'word_stress_features.json')
    word_df.to_json(json_path, orient='records', indent=2)
    print(f"Word-level features (JSON) saved to: {json_path}")

    # Generate visualizations (using primary stress_label)
    print("\nGenerating visualizations...")
    visualize_stress_clusters(word_df, args.output_dir)

    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    if len(all_metrics) > 1:
        print(f"  - Comparison: {metrics_path}")
    print(f"  - Visualizations: {args.output_dir}/word_stress_*.png")
    samples = word_df[word_df['stress_label']
                      == stress]['word'].head(10).tolist()
    print(f"  {stress.upper()}: {', '.join(samples)}")

    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_stress_clusters(word_df, args.output_dir)

    print("\n" + "="*70)
    print("CLUSTERING COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Visualizations: {args.output_dir}/word_stress_*.png")


if __name__ == '__main__':
    main()
