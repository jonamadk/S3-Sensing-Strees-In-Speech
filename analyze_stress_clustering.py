#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis and Visualization for Word Stress Clustering
Generates detailed reports, charts, and statistical tests to validate clustering quality.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data(csv_path):
    """Load clustered word stress data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} words from {csv_path}")
    return df


def vowel_distribution_analysis(df, output_dir):
    """Analyze and visualize vowel distributions across stress levels."""
    print("\n" + "="*80)
    print("VOWEL DISTRIBUTION ANALYSIS")
    print("="*80)

    # Extract individual vowels from phoneme strings
    VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
    VOWELS.update({"o", "ᵻ", "a", "e", "ɐ", "ɒ", "ɜ"})

    # Also handle vowels with stress markers (AH0, AH1, etc.)
    def extract_vowels(phonemes):
        if pd.isna(phonemes):
            return []
        vowels = []
        i = 0
        while i < len(phonemes):
            # Try 3-char match (e.g., AH0)
            if i + 3 <= len(phonemes):
                three_char = phonemes[i:i+3]
                base = ''.join([c for c in three_char if not c.isdigit()])
                if base in VOWELS:
                    vowels.append(three_char)
                    i += 3
                    continue
            # Try 2-char match
            if i + 2 <= len(phonemes):
                two_char = phonemes[i:i+2]
                if two_char in VOWELS:
                    vowels.append(two_char)
                    i += 2
                    continue
            # Try 1-char match
            if phonemes[i] in VOWELS:
                vowels.append(phonemes[i])
                i += 1
                continue
            i += 1
        return vowels

    df['vowel_list'] = df['phonemes'].apply(extract_vowels)
    df['num_vowels_extracted'] = df['vowel_list'].apply(len)

    # Count vowel types by stress level
    vowel_counts = {}
    for stress in df['stress_label'].unique():
        stress_df = df[df['stress_label'] == stress]
        all_vowels = []
        for v_list in stress_df['vowel_list']:
            all_vowels.extend(v_list)
        vowel_counts[stress] = pd.Series(all_vowels).value_counts()

    # Create vowel distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Vowel count distribution
    ax1 = axes[0, 0]
    df.boxplot(column='num_vowels', by='stress_label', ax=ax1)
    ax1.set_title('Number of Vowels per Word by Stress Level',
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Stress Level', fontsize=11)
    ax1.set_ylabel('Number of Vowels', fontsize=11)
    plt.sca(ax1)
    plt.xticks(rotation=0)

    # 2. Top 10 vowels by stress level
    ax2 = axes[0, 1]
    top_vowels = set()
    for counts in vowel_counts.values():
        top_vowels.update(counts.head(10).index)

    vowel_data = []
    for vowel in sorted(top_vowels):
        for stress in df['stress_label'].unique():
            count = vowel_counts[stress].get(vowel, 0)
            vowel_data.append(
                {'Vowel': vowel, 'Stress': stress, 'Count': count})

    vowel_df = pd.DataFrame(vowel_data)
    vowel_pivot = vowel_df.pivot(
        index='Vowel', columns='Stress', values='Count').fillna(0)
    vowel_pivot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Top Vowel Frequencies by Stress Level',
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel('Vowel Phoneme', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(title='Stress Level')
    plt.sca(ax2)
    plt.xticks(rotation=45)

    # 3. Vowel ratio distribution
    ax3 = axes[1, 0]
    for stress in df['stress_label'].unique():
        data = df[df['stress_label'] == stress]['vowel_ratio']
        ax3.hist(data, alpha=0.6, bins=20, label=stress.capitalize())
    ax3.set_title('Vowel Ratio Distribution by Stress Level',
                  fontweight='bold', fontsize=12)
    ax3.set_xlabel('Vowel Ratio (vowel_duration / word_duration)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Vowel duration vs word duration
    ax4 = axes[1, 1]
    for stress in df['stress_label'].unique():
        data = df[df['stress_label'] == stress]
        ax4.scatter(data['word_duration'], data['vowel_duration'],
                    alpha=0.5, label=stress.capitalize(), s=50)
    ax4.set_title('Vowel Duration vs Word Duration',
                  fontweight='bold', fontsize=12)
    ax4.set_xlabel('Word Duration (s)', fontsize=11)
    ax4.set_ylabel('Vowel Duration (s)', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'vowel_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Vowel distribution visualization saved to: {output_path}")

    # Print statistics
    print("\nVowel Statistics by Stress Level:")
    print(df.groupby('stress_label')[
          ['num_vowels', 'vowel_ratio', 'vowel_duration']].describe().round(3))

    return vowel_counts


def prosodic_features_analysis(df, output_dir):
    """Analyze relationships between prosodic features."""
    print("\n" + "="*80)
    print("PROSODIC FEATURES CORRELATION ANALYSIS")
    print("="*80)

    # Select key prosodic features
    features = ['vowel_duration', 'vowel_ratio', 'pitch_mean', 'pitch_max',
                'pitch_range', 'pitch_slope', 'energy_max', 'word_duration', 'prominence_score']

    # Filter features that exist
    features = [f for f in features if f in df.columns]

    # Correlation matrix
    corr_matrix = df[features].corr()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Correlation heatmap
    ax1 = axes[0, 0]
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Prosodic Feature Correlation Matrix',
                  fontweight='bold', fontsize=12)
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 2. Pitch vs Duration by stress
    ax2 = axes[0, 1]
    for stress in df['stress_label'].unique():
        data = df[df['stress_label'] == stress]
        ax2.scatter(data['word_duration'], data['pitch_max'],
                    alpha=0.6, label=stress.capitalize(), s=80)
    ax2.set_title('Pitch Max vs Duration by Stress',
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel('Word Duration (s)', fontsize=11)
    ax2.set_ylabel('Max Pitch (Hz)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Energy vs Pitch
    ax3 = axes[1, 0]
    for stress in df['stress_label'].unique():
        data = df[df['stress_label'] == stress]
        ax3.scatter(data['pitch_max'], data['energy_max'],
                    alpha=0.6, label=stress.capitalize(), s=80)
    ax3.set_title('Energy vs Pitch by Stress', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Max Pitch (Hz)', fontsize=11)
    ax3.set_ylabel('Max Energy (RMS)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Prominence score distribution
    ax4 = axes[1, 1]
    df.boxplot(column='prominence_score', by='stress_label', ax=ax4)
    ax4.set_title('Prominence Score Distribution by Stress',
                  fontweight='bold', fontsize=12)
    ax4.set_xlabel('Stress Level', fontsize=11)
    ax4.set_ylabel('Prominence Score', fontsize=11)
    plt.sca(ax4)
    plt.xticks(rotation=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prosodic_features_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prosodic features analysis saved to: {output_path}")

    # Print top correlations
    print("\nTop Feature Correlations:")
    corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_pairs.append(
                (features[i], features[j], corr_matrix.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr in corr_pairs[:10]:
        print(f"  {feat1:20s} <-> {feat2:20s}: {corr:6.3f}")


def statistical_significance_tests(df):
    """Perform statistical tests to validate clustering quality."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    # Features to test
    test_features = ['vowel_duration', 'vowel_ratio', 'pitch_max', 'pitch_range',
                     'energy_max', 'prominence_score', 'word_duration']
    test_features = [f for f in test_features if f in df.columns]

    results = []

    for feature in test_features:
        # Group data by stress level
        groups = [df[df['stress_label'] == stress][feature].dropna()
                  for stress in df['stress_label'].unique()]

        # ANOVA test (parametric)
        f_stat, p_value_anova = f_oneway(*groups)

        # Kruskal-Wallis test (non-parametric)
        h_stat, p_value_kruskal = kruskal(*groups)

        results.append({
            'Feature': feature,
            'ANOVA F-stat': f_stat,
            'ANOVA p-value': p_value_anova,
            'Kruskal H-stat': h_stat,
            'Kruskal p-value': p_value_kruskal,
            'Significant (p<0.05)': 'Yes' if p_value_anova < 0.05 else 'No'
        })

    results_df = pd.DataFrame(results)
    print("\nANOVA and Kruskal-Wallis Tests (comparing stress levels):")
    print(results_df.to_string(index=False))

    # Interpretation
    print("\nInterpretation:")
    print("  - p-value < 0.05: Feature significantly differs across stress levels")
    print("  - ANOVA: Assumes normal distribution")
    print("  - Kruskal-Wallis: Non-parametric alternative")

    # Pairwise comparisons for vowel_duration
    if 'vowel_duration' in df.columns:
        print("\n" + "-"*80)
        print("Pairwise T-Tests for Vowel Duration:")
        print("-"*80)
        stress_levels = df['stress_label'].unique()
        for i, s1 in enumerate(stress_levels):
            for s2 in stress_levels[i+1:]:
                g1 = df[df['stress_label'] == s1]['vowel_duration'].dropna()
                g2 = df[df['stress_label'] == s2]['vowel_duration'].dropna()
                t_stat, p_val = stats.ttest_ind(g1, g2)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(
                    f"  {s1:12s} vs {s2:12s}: t={t_stat:7.3f}, p={p_val:.4e} {sig}")

    return results_df


def clustering_quality_metrics(df, output_dir):
    """Calculate clustering quality metrics."""
    print("\n" + "="*80)
    print("CLUSTERING QUALITY METRICS")
    print("="*80)

    # Select features used for clustering
    feature_cols = ['vowel_duration', 'vowel_ratio', 'pitch_mean', 'pitch_max',
                    'pitch_range', 'pitch_slope', 'pitch_std', 'pre_pause', 'prominence_score']
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols].fillna(0).values
    labels = df['cluster'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)

    print(
        f"\nSilhouette Score:        {silhouette:.4f}  (range: -1 to 1, higher is better)")
    print(
        f"Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better, 0 is best)")
    print(f"Calinski-Harabasz Score: {calinski:.4f}  (higher is better)")

    # Interpretation
    print("\nInterpretation:")
    if silhouette > 0.5:
        print("  ✓ Silhouette score > 0.5: Good cluster separation")
    elif silhouette > 0.25:
        print("  ✓ Silhouette score > 0.25: Reasonable cluster structure")
    else:
        print("  ⚠ Silhouette score < 0.25: Weak cluster structure")

    # Cluster size balance
    cluster_sizes = df['cluster'].value_counts().sort_index()
    print(f"\nCluster Sizes:")
    for cluster, size in cluster_sizes.items():
        stress = df[df['cluster'] == cluster]['stress_label'].iloc[0]
        print(
            f"  Cluster {cluster} ({stress:12s}): {size:4d} words ({100*size/len(df):5.2f}%)")

    # Visualize cluster separation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Silhouette plot
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X_scaled, labels)

    ax1 = axes[0]
    y_lower = 10
    for i in sorted(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        stress_label = df[df['cluster'] == i]['stress_label'].iloc[0]
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                          label=f'Cluster {i} ({stress_label})', alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title('Silhouette Plot for Clustering',
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax1.set_ylabel('Cluster', fontsize=11)
    ax1.axvline(x=silhouette, color="red", linestyle="--",
                label=f'Average: {silhouette:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Feature importance for clustering
    ax2 = axes[1]
    # Calculate variance of each feature across clusters
    feature_importance = []
    for feat in feature_cols:
        variance = df.groupby('cluster')[feat].var().mean()
        feature_importance.append({'Feature': feat, 'Variance': variance})

    feat_imp_df = pd.DataFrame(feature_importance).sort_values(
        'Variance', ascending=False)
    ax2.barh(feat_imp_df['Feature'], feat_imp_df['Variance'])
    ax2.set_title('Feature Variance Across Clusters',
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel('Mean Variance', fontsize=11)
    ax2.set_ylabel('Feature', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'clustering_quality_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Clustering quality visualization saved to: {output_path}")

    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski
    }


def advanced_visualizations(df, output_dir):
    """Create advanced multi-dimensional visualizations."""
    print("\n" + "="*80)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("="*80)

    # 1. Pair plot for key features
    print("Creating pair plot...")
    key_features = ['vowel_duration', 'vowel_ratio',
                    'pitch_max', 'prominence_score', 'stress_label']
    key_features = [f for f in key_features if f in df.columns]

    pairplot = sns.pairplot(df[key_features], hue='stress_label',
                            diag_kind='kde', plot_kws={'alpha': 0.6},
                            height=3, aspect=1)
    pairplot.fig.suptitle('Pairwise Feature Relationships by Stress Level',
                          y=1.02, fontsize=14, fontweight='bold')
    output_path = os.path.join(output_dir, 'feature_pairplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Pair plot saved to: {output_path}")
    plt.close()

    # 2. Ridge plots for distributions
    print("Creating ridge plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    features_to_plot = ['vowel_duration', 'pitch_max',
                        'pitch_range', 'prominence_score']
    features_to_plot = [f for f in features_to_plot if f in df.columns]

    for i, feature in enumerate(features_to_plot[:4]):
        ax = axes[i]
        for stress in df['stress_label'].unique():
            data = df[df['stress_label'] == stress][feature].dropna()
            ax.hist(data, alpha=0.5, bins=30,
                    label=stress.capitalize(), density=True)

        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Distribution Comparison by Stress Level',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plots saved to: {output_path}")
    plt.close()

    # 3. 3D scatter plot
    print("Creating 3D visualization...")
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    stress_colors = {stress: i for i,
                     stress in enumerate(df['stress_label'].unique())}
    colors = df['stress_label'].map(stress_colors)

    scatter = ax.scatter(df['vowel_duration'], df['pitch_max'], df['prominence_score'],
                         c=colors, cmap='viridis', s=50, alpha=0.6)

    ax.set_xlabel('Vowel Duration (s)', fontsize=11)
    ax.set_ylabel('Max Pitch (Hz)', fontsize=11)
    ax.set_zlabel('Prominence Score', fontsize=11)
    ax.set_title('3D Feature Space: Duration × Pitch × Prominence',
                 fontsize=13, fontweight='bold')

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.viridis(
                              stress_colors[s]/len(stress_colors)),
                          markersize=8, label=s.capitalize())
               for s in df['stress_label'].unique()]
    ax.legend(handles=handles, loc='upper right')

    output_path = os.path.join(output_dir, 'feature_3d_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3D plot saved to: {output_path}")
    plt.close()


def generate_summary_report(df, stats_results, quality_metrics, output_dir):
    """Generate comprehensive text report."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    report_path = os.path.join(output_dir, 'clustering_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VOWEL-BASED WORD STRESS CLUSTERING - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Dataset: {len(df)} words\n")
        f.write(
            f"Features analyzed: vowel duration, pitch dynamics, energy, prominence\n\n")

        # Cluster summary
        f.write("-"*80 + "\n")
        f.write("CLUSTER DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        cluster_summary = df['stress_label'].value_counts()
        for stress, count in cluster_summary.items():
            f.write(
                f"{stress.upper():15s}: {count:4d} words ({100*count/len(df):5.2f}%)\n")

        # Feature statistics
        f.write("\n" + "-"*80 + "\n")
        f.write("FEATURE STATISTICS BY STRESS LEVEL\n")
        f.write("-"*80 + "\n")
        feature_stats = df.groupby('stress_label')[
            ['vowel_duration', 'vowel_ratio', 'pitch_max', 'pitch_range',
             'energy_max', 'prominence_score']
        ].mean()
        f.write(feature_stats.to_string())
        f.write("\n")

        # Statistical significance
        f.write("\n" + "-"*80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-"*80 + "\n")
        f.write(stats_results.to_string(index=False))
        f.write("\n")

        # Clustering quality
        f.write("\n" + "-"*80 + "\n")
        f.write("CLUSTERING QUALITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(
            f"Silhouette Score:        {quality_metrics['silhouette']:.4f}\n")
        f.write(
            f"Davies-Bouldin Index:    {quality_metrics['davies_bouldin']:.4f}\n")
        f.write(
            f"Calinski-Harabasz Score: {quality_metrics['calinski_harabasz']:.4f}\n")

        # Key findings
        f.write("\n" + "-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")

        # Vowel duration analysis
        vd_by_stress = df.groupby('stress_label')['vowel_duration'].mean()
        if len(vd_by_stress) == 3:
            primary_vd = vd_by_stress.get('primary', 0)
            unstressed_vd = vd_by_stress.get('unstressed', 0)
            ratio = primary_vd / unstressed_vd if unstressed_vd > 0 else 0
            f.write(
                f"1. Primary stressed words have {ratio:.2f}x longer vowels than unstressed\n")

        # Pitch analysis
        pitch_by_stress = df.groupby('stress_label')['pitch_max'].mean()
        if len(pitch_by_stress) >= 2:
            max_stress = pitch_by_stress.idxmax()
            min_stress = pitch_by_stress.idxmin()
            diff = pitch_by_stress[max_stress] - pitch_by_stress[min_stress]
            f.write(
                f"2. {max_stress.capitalize()} words have {diff:.1f} Hz higher pitch than {min_stress}\n")

        # Significance
        sig_features = stats_results[stats_results['ANOVA p-value']
                                     < 0.05]['Feature'].tolist()
        f.write(
            f"3. Statistically significant features (p<0.05): {', '.join(sig_features)}\n")

        # Clustering quality
        if quality_metrics['silhouette'] > 0.5:
            f.write(
                f"4. Strong cluster separation (silhouette={quality_metrics['silhouette']:.3f})\n")
        elif quality_metrics['silhouette'] > 0.25:
            f.write(
                f"4. Moderate cluster separation (silhouette={quality_metrics['silhouette']:.3f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary report saved to: {report_path}")

    # Also print to console
    with open(report_path, 'r') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of word stress clustering with statistical tests')
    parser.add_argument('--input-csv', required=True,
                        help='Input CSV file with clustered word features')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for visualizations and reports')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("VOWEL-BASED STRESS CLUSTERING - COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load data
    df = load_data(args.input_csv)

    # Run analyses
    vowel_counts = vowel_distribution_analysis(df, args.output_dir)
    prosodic_features_analysis(df, args.output_dir)
    stats_results = statistical_significance_tests(df)
    quality_metrics = clustering_quality_metrics(df, args.output_dir)
    advanced_visualizations(df, args.output_dir)
    generate_summary_report(
        df, stats_results, quality_metrics, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - vowel_distribution_analysis.png")
    print("  - prosodic_features_analysis.png")
    print("  - clustering_quality_metrics.png")
    print("  - feature_pairplot.png")
    print("  - feature_distributions.png")
    print("  - feature_3d_plot.png")
    print("  - clustering_analysis_report.txt")


if __name__ == '__main__':
    main()
