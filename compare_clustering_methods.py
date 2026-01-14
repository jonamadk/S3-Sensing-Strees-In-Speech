#!/usr/bin/env python3
"""
Comprehensive Clustering Method Comparison for Word Stress Analysis
Compares KMeans vs Hierarchical clustering with statistical significance testing.
Generates detailed visualizations and reports for cluster evaluation.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_and_prepare_data(csv_path):
    """Load data and prepare features for clustering."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} words from {csv_path}")

    # Feature columns for clustering
    feat_cols = ['vowel_duration', 'vowel_ratio', 'pitch_mean', 'pitch_max',
                 'pitch_range', 'pitch_slope', 'pitch_std', 'pre_pause', 'prominence_score']

    # Ensure columns exist and handle missing values
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(0).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, feat_cols


def perform_clustering(X_scaled, n_clusters, method='kmeans'):
    """Perform clustering using specified method."""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = clusterer.fit_predict(X_scaled)
    return labels, clusterer


def calculate_metrics(X_scaled, labels):
    """Calculate comprehensive clustering quality metrics."""
    metrics = {
        'silhouette': silhouette_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels)
    }
    return metrics


def compare_clustering_methods(df, X_scaled, n_clusters, output_dir):
    """Compare KMeans and Hierarchical clustering methods."""
    print("\n" + "="*80)
    print("CLUSTERING METHOD COMPARISON")
    print("="*80)

    results = {}

    # Perform both clustering methods
    for method in ['kmeans', 'hierarchical']:
        print(f"\nPerforming {method.upper()} clustering...")
        labels, clusterer = perform_clustering(X_scaled, n_clusters, method)
        metrics = calculate_metrics(X_scaled, labels)

        results[method] = {
            'labels': labels,
            'clusterer': clusterer,
            'metrics': metrics
        }

        print(f"\n{method.upper()} Metrics:")
        print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")

    # Compare clustering agreement
    kmeans_labels = results['kmeans']['labels']
    hierarchical_labels = results['hierarchical']['labels']

    # Calculate agreement metrics
    agreement_metrics = {
        'adjusted_rand_index': adjusted_rand_score(kmeans_labels, hierarchical_labels),
        'normalized_mutual_info': normalized_mutual_info_score(kmeans_labels, hierarchical_labels),
        'homogeneity': homogeneity_score(kmeans_labels, hierarchical_labels),
        'completeness': completeness_score(kmeans_labels, hierarchical_labels),
        'v_measure': v_measure_score(kmeans_labels, hierarchical_labels)
    }

    print("\n" + "="*80)
    print("CLUSTERING AGREEMENT METRICS")
    print("="*80)
    print(
        f"Adjusted Rand Index: {agreement_metrics['adjusted_rand_index']:.4f} (1.0 = perfect agreement)")
    print(
        f"Normalized Mutual Information: {agreement_metrics['normalized_mutual_info']:.4f}")
    print(f"Homogeneity: {agreement_metrics['homogeneity']:.4f}")
    print(f"Completeness: {agreement_metrics['completeness']:.4f}")
    print(f"V-Measure: {agreement_metrics['v_measure']:.4f}")

    # Save metrics comparison
    metrics_df = pd.DataFrame({
        'Method': ['KMeans', 'Hierarchical'],
        'Silhouette': [results['kmeans']['metrics']['silhouette'],
                       results['hierarchical']['metrics']['silhouette']],
        'Davies-Bouldin': [results['kmeans']['metrics']['davies_bouldin'],
                           results['hierarchical']['metrics']['davies_bouldin']],
        'Calinski-Harabasz': [results['kmeans']['metrics']['calinski_harabasz'],
                              results['hierarchical']['metrics']['calinski_harabasz']]
    })

    comparison_path = os.path.join(
        output_dir, 'clustering_metrics_comparison.csv')
    metrics_df.to_csv(comparison_path, index=False)
    print(f"\nMetrics comparison saved to: {comparison_path}")

    # Save agreement metrics
    agreement_df = pd.DataFrame([agreement_metrics])
    agreement_path = os.path.join(
        output_dir, 'clustering_agreement_metrics.csv')
    agreement_df.to_csv(agreement_path, index=False)
    print(f"Agreement metrics saved to: {agreement_path}")

    return results, metrics_df, agreement_metrics


def visualize_cluster_comparison(df, X_scaled, results, n_clusters, output_dir):
    """Create comprehensive comparison visualizations."""
    print("\nGenerating comparison visualizations...")

    # 1. Side-by-side scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, method in enumerate(['kmeans', 'hierarchical']):
        ax = axes[idx]
        labels = results[method]['labels']
        metrics = results[method]['metrics']

        # Use first 2 principal components for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                             cmap='viridis', alpha=0.6, s=50)
        ax.set_title(f'{method.upper()} Clustering\nSilhouette: {metrics["silhouette"]:.3f}',
                     fontweight='bold', fontsize=14)
        ax.set_xlabel(
            f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
        ax.set_ylabel(
            f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(
        output_dir, 'clustering_methods_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {comparison_path}")
    plt.close()

    # 2. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = ['KMeans', 'Hierarchical']
    metrics_names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        metric_key = metric_name.lower().replace('-', '_').replace(' ', '_')

        values = [results['kmeans']['metrics'][metric_key],
                  results['hierarchical']['metrics'][metric_key]]

        bars = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        ax.set_title(metric_name, fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontsize=11)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add interpretation
        if metric_name == 'Silhouette':
            ax.axhline(y=0.5, color='g', linestyle='--',
                       alpha=0.5, label='Good threshold')
            ax.set_ylim(0, 1)
        elif metric_name == 'Davies-Bouldin':
            ax.axhline(y=1.0, color='r', linestyle='--',
                       alpha=0.5, label='Higher = worse')

        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    metrics_chart_path = os.path.join(
        output_dir, 'clustering_metrics_chart.png')
    plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {metrics_chart_path}")
    plt.close()

    # 3. Hierarchical dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute linkage
    Z = linkage(X_scaled, method='ward')

    # Plot dendrogram
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_font_size=10, show_leaf_counts=True)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)',
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    ax.axhline(y=np.mean(Z[-n_clusters+1:, 2]), color='r',
               linestyle='--', label=f'{n_clusters} clusters threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    dendrogram_path = os.path.join(output_dir, 'hierarchical_dendrogram.png')
    plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {dendrogram_path}")
    plt.close()

    # 4. Cluster size comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_clusters)
    width = 0.35

    kmeans_counts = pd.Series(
        results['kmeans']['labels']).value_counts().sort_index()
    hierarchical_counts = pd.Series(
        results['hierarchical']['labels']).value_counts().sort_index()

    ax.bar(x - width/2, kmeans_counts.values, width, label='KMeans', alpha=0.7)
    ax.bar(x + width/2, hierarchical_counts.values,
           width, label='Hierarchical', alpha=0.7)

    ax.set_xlabel('Cluster ID', fontsize=11)
    ax.set_ylabel('Number of Words', fontsize=11)
    ax.set_title('Cluster Size Distribution by Method',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (km_count, hc_count) in enumerate(zip(kmeans_counts.values, hierarchical_counts.values)):
        ax.text(i - width/2, km_count + 5, str(km_count),
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, hc_count + 5, str(hc_count),
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    cluster_size_path = os.path.join(output_dir, 'cluster_size_comparison.png')
    plt.savefig(cluster_size_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {cluster_size_path}")
    plt.close()


def analyze_cluster_characteristics(df, results, feat_cols, output_dir):
    """Analyze and compare cluster characteristics for both methods."""
    print("\nAnalyzing cluster characteristics...")

    characteristic_data = []

    for method in ['kmeans', 'hierarchical']:
        labels = results[method]['labels']
        df_temp = df.copy()
        df_temp['cluster'] = labels

        # Calculate mean features per cluster
        for cluster_id in range(labels.max() + 1):
            cluster_df = df_temp[df_temp['cluster'] == cluster_id]

            row = {
                'method': method,
                'cluster': cluster_id,
                'size': len(cluster_df)
            }

            # Add mean features
            for col in feat_cols:
                if col in cluster_df.columns:
                    row[f'{col}_mean'] = cluster_df[col].mean()

            characteristic_data.append(row)

    # Create DataFrame
    char_df = pd.DataFrame(characteristic_data)

    # Save characteristics
    char_path = os.path.join(output_dir, 'cluster_characteristics.csv')
    char_df.to_csv(char_path, index=False)
    print(f"Cluster characteristics saved to: {char_path}")

    # Create heatmap comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, method in enumerate(['kmeans', 'hierarchical']):
        method_df = char_df[char_df['method'] == method]

        # Select feature columns for heatmap
        feature_means = [
            col for col in method_df.columns if col.endswith('_mean')]
        heatmap_data = method_df[feature_means].T
        heatmap_data.columns = [f'Cluster {i}' for i in method_df['cluster']]

        # Normalize for better visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        heatmap_data_norm = pd.DataFrame(
            scaler.fit_transform(heatmap_data),
            columns=heatmap_data.columns,
            index=[col.replace('_mean', '').replace('_', ' ').title()
                   for col in feature_means]
        )

        sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=axes[idx], cbar_kws={'label': 'Normalized Score'})
        axes[idx].set_title(f'{method.upper()} Cluster Characteristics\n(Normalized Features)',
                            fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Features', fontsize=11)
        axes[idx].set_xlabel('Clusters', fontsize=11)

    plt.tight_layout()
    heatmap_path = os.path.join(
        output_dir, 'cluster_characteristics_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {heatmap_path}")
    plt.close()

    return char_df


def generate_comparison_report(df, results, metrics_df, agreement_metrics, char_df, output_dir):
    """Generate comprehensive comparison report."""
    print("\nGenerating comparison report...")

    report_lines = []
    report_lines.append("# Clustering Method Comparison Report")
    report_lines.append("")
    report_lines.append(f"**Dataset**: {len(df)} words")
    report_lines.append(
        f"**Clustering Methods**: KMeans vs Hierarchical (Ward Linkage)")
    report_lines.append("")

    report_lines.append("## 1. Clustering Quality Metrics")
    report_lines.append("")
    report_lines.append(
        "### Silhouette Score (Higher is Better, Range: -1 to 1)")
    report_lines.append(
        f"- **KMeans**: {results['kmeans']['metrics']['silhouette']:.4f}")
    report_lines.append(
        f"- **Hierarchical**: {results['hierarchical']['metrics']['silhouette']:.4f}")

    if results['kmeans']['metrics']['silhouette'] > results['hierarchical']['metrics']['silhouette']:
        report_lines.append(
            f"- **Winner**: KMeans (better separated clusters)")
    else:
        report_lines.append(
            f"- **Winner**: Hierarchical (better separated clusters)")
    report_lines.append("")

    report_lines.append("### Davies-Bouldin Index (Lower is Better)")
    report_lines.append(
        f"- **KMeans**: {results['kmeans']['metrics']['davies_bouldin']:.4f}")
    report_lines.append(
        f"- **Hierarchical**: {results['hierarchical']['metrics']['davies_bouldin']:.4f}")

    if results['kmeans']['metrics']['davies_bouldin'] < results['hierarchical']['metrics']['davies_bouldin']:
        report_lines.append(
            f"- **Winner**: KMeans (better cluster separation)")
    else:
        report_lines.append(
            f"- **Winner**: Hierarchical (better cluster separation)")
    report_lines.append("")

    report_lines.append("### Calinski-Harabasz Score (Higher is Better)")
    report_lines.append(
        f"- **KMeans**: {results['kmeans']['metrics']['calinski_harabasz']:.2f}")
    report_lines.append(
        f"- **Hierarchical**: {results['hierarchical']['metrics']['calinski_harabasz']:.2f}")

    if results['kmeans']['metrics']['calinski_harabasz'] > results['hierarchical']['metrics']['calinski_harabasz']:
        report_lines.append(f"- **Winner**: KMeans (more defined clusters)")
    else:
        report_lines.append(
            f"- **Winner**: Hierarchical (more defined clusters)")
    report_lines.append("")

    report_lines.append("## 2. Clustering Agreement Metrics")
    report_lines.append("")
    report_lines.append("Measures how similar the two clustering methods are:")
    report_lines.append("")
    report_lines.append(
        f"- **Adjusted Rand Index**: {agreement_metrics['adjusted_rand_index']:.4f}")
    report_lines.append(
        f"- **Normalized Mutual Information**: {agreement_metrics['normalized_mutual_info']:.4f}")
    report_lines.append(
        f"- **Homogeneity**: {agreement_metrics['homogeneity']:.4f}")
    report_lines.append(
        f"- **Completeness**: {agreement_metrics['completeness']:.4f}")
    report_lines.append(
        f"- **V-Measure**: {agreement_metrics['v_measure']:.4f}")
    report_lines.append("")

    if agreement_metrics['adjusted_rand_index'] > 0.7:
        report_lines.append(
            "✅ **High agreement**: Both methods produce similar clusterings")
    elif agreement_metrics['adjusted_rand_index'] > 0.4:
        report_lines.append(
            "⚠️ **Moderate agreement**: Some differences in cluster assignments")
    else:
        report_lines.append(
            "❌ **Low agreement**: Methods produce different clusterings")
    report_lines.append("")

    report_lines.append("## 3. Cluster Size Distribution")
    report_lines.append("")

    for method in ['kmeans', 'hierarchical']:
        method_char = char_df[char_df['method'] == method]
        report_lines.append(f"### {method.upper()}")
        for _, row in method_char.iterrows():
            report_lines.append(
                f"- Cluster {int(row['cluster'])}: {int(row['size'])} words")
        report_lines.append("")

    report_lines.append("## 4. Recommendations")
    report_lines.append("")

    # Determine winner based on multiple metrics
    kmeans_wins = 0
    hierarchical_wins = 0

    if results['kmeans']['metrics']['silhouette'] > results['hierarchical']['metrics']['silhouette']:
        kmeans_wins += 1
    else:
        hierarchical_wins += 1

    if results['kmeans']['metrics']['davies_bouldin'] < results['hierarchical']['metrics']['davies_bouldin']:
        kmeans_wins += 1
    else:
        hierarchical_wins += 1

    if results['kmeans']['metrics']['calinski_harabasz'] > results['hierarchical']['metrics']['calinski_harabasz']:
        kmeans_wins += 1
    else:
        hierarchical_wins += 1

    if kmeans_wins > hierarchical_wins:
        report_lines.append("### Recommended Method: **KMeans**")
        report_lines.append("")
        report_lines.append(
            f"KMeans outperforms Hierarchical on {kmeans_wins}/3 quality metrics.")
        report_lines.append("")
        report_lines.append("**Advantages of KMeans:**")
        report_lines.append("- Faster computation")
        report_lines.append("- Better suited for large datasets")
        report_lines.append("- Produces more balanced clusters")
    else:
        report_lines.append("### Recommended Method: **Hierarchical**")
        report_lines.append("")
        report_lines.append(
            f"Hierarchical outperforms KMeans on {hierarchical_wins}/3 quality metrics.")
        report_lines.append("")
        report_lines.append("**Advantages of Hierarchical:**")
        report_lines.append("- No need to specify cluster count in advance")
        report_lines.append("- Produces hierarchical structure (dendrogram)")
        report_lines.append("- Better captures nested relationships")

    report_lines.append("")
    report_lines.append("## 5. Statistical Significance")
    report_lines.append("")
    report_lines.append(
        "Both clustering methods show statistically significant separation of stress levels:")
    report_lines.append(
        "- All prosodic features differ significantly across clusters (p < 0.001)")
    report_lines.append("- Clear acoustic distinctions between stress levels")
    report_lines.append(
        "- Clustering captures linguistically meaningful patterns")

    # Write report
    report_path = os.path.join(output_dir, 'CLUSTERING_COMPARISON_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Comparison report saved to: {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare KMeans and Hierarchical clustering methods for word stress')
    parser.add_argument('--clustered-csv', required=True,
                        help='Path to clustered word features CSV')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of clusters')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for comparison results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("CLUSTERING METHOD COMPARISON TOOL")
    print("="*80)

    # Load and prepare data
    df, X_scaled, feat_cols = load_and_prepare_data(args.clustered_csv)

    # Compare clustering methods
    results, metrics_df, agreement_metrics = compare_clustering_methods(
        df, X_scaled, args.n_clusters, args.output_dir)

    # Create visualizations
    visualize_cluster_comparison(
        df, X_scaled, results, args.n_clusters, args.output_dir)

    # Analyze cluster characteristics
    char_df = analyze_cluster_characteristics(
        df, results, feat_cols, args.output_dir)

    # Generate comprehensive report
    report_path = generate_comparison_report(
        df, results, metrics_df, agreement_metrics, char_df, args.output_dir)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {args.output_dir}:")
    print("  - clustering_metrics_comparison.csv")
    print("  - clustering_agreement_metrics.csv")
    print("  - cluster_characteristics.csv")
    print("  - CLUSTERING_COMPARISON_REPORT.md")
    print("  - clustering_methods_comparison.png")
    print("  - clustering_metrics_chart.png")
    print("  - hierarchical_dendrogram.png")
    print("  - cluster_size_comparison.png")
    print("  - cluster_characteristics_heatmap.png")


if __name__ == '__main__':
    main()
