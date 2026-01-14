# Clustering Methods Comparison Summary

## Overview

This document summarizes the comparison between **KMeans** and **Hierarchical clustering** methods for word stress detection, evaluated on both 3-class and 2-class clustering tasks.

---

## 📊 Clustering Quality Metrics

### 3-Class Clustering (Primary / Secondary / Unstressed)

| Method | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|------------|----------------|-------------------|
| **KMeans** | **0.2819** | 1.2346 | **423.55** |
| **Hierarchical** | 0.2701 | **0.9865** | 356.27 |

**Winner**: **KMeans** (2/3 metrics)

### 2-Class Clustering (Stressed / Unstressed)

| Method | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|------------|----------------|-------------------|
| **KMeans** | 0.6034 | 0.9577 | **393.50** |
| **Hierarchical** | **0.6462** | **0.8254** | 363.72 |

**Winner**: **Hierarchical** (2/3 metrics)

---

## 🤝 Clustering Agreement Analysis

Measures how similar the two methods' cluster assignments are:

### 3-Class Clustering
- **Adjusted Rand Index**: 0.1573 (LOW agreement)
- **Normalized Mutual Info**: 0.3598
- **V-Measure**: 0.3598
- **Interpretation**: ❌ Methods produce different clusterings

### 2-Class Clustering
- **Adjusted Rand Index**: 0.7795 (HIGH agreement)
- **Normalized Mutual Info**: 0.6649
- **V-Measure**: 0.6649
- **Interpretation**: ✅ Methods largely agree on cluster assignments

---

## 📈 Cluster Size Distribution

### 3-Class Clustering

**KMeans:**
- Cluster 0 (Unstressed): 564 words (53.7%)
- Cluster 1 (Secondary): 431 words (41.0%)
- Cluster 2 (Primary): 56 words (5.3%)

**Hierarchical:**
- Cluster 0 (Secondary): 793 words (75.5%)
- Cluster 1 (Primary): 44 words (4.2%)
- Cluster 2 (Unstressed): 214 words (20.4%)

**Observation**: KMeans produces more balanced clusters

### 2-Class Clustering

**KMeans:**
- Cluster 0 (Unstressed): 986 words (93.8%)
- Cluster 1 (Stressed): 65 words (6.2%)

**Hierarchical:**
- Cluster 0 (Unstressed): 1007 words (95.8%)
- Cluster 1 (Stressed): 44 words (4.2%)

**Observation**: Both methods agree on high imbalance (most words unstressed)

---

## 🎯 Key Findings

### Metric Interpretations

1. **Silhouette Score** (range: -1 to 1, higher is better)
   - Measures how well each point fits in its cluster vs neighbors
   - **3-class**: Both methods show moderate separation (0.27-0.28)
   - **2-class**: Both methods show good separation (0.60-0.65)
   - **Conclusion**: Binary classification is much clearer

2. **Davies-Bouldin Index** (lower is better)
   - Measures average similarity ratio between clusters
   - **3-class**: Hierarchical wins (0.99 vs 1.23)
   - **2-class**: Hierarchical wins (0.83 vs 0.96)
   - **Conclusion**: Hierarchical produces tighter clusters

3. **Calinski-Harabasz Score** (higher is better)
   - Ratio of between-cluster to within-cluster variance
   - **3-class**: KMeans wins (423 vs 356)
   - **2-class**: KMeans wins (393 vs 364)
   - **Conclusion**: KMeans separates clusters more distinctly

### Statistical Significance

Both methods show **statistically significant** separation:
- All prosodic features differ across clusters (p < 0.001)
- Clear acoustic distinctions between stress levels
- Clustering captures linguistically meaningful patterns

---

## 💡 Recommendations

### For 3-Class Task (Primary/Secondary/Unstressed):
**→ Use KMeans Clustering**

**Reasons:**
- ✅ Better Silhouette score (0.2819 vs 0.2701)
- ✅ Higher Calinski-Harabasz (423 vs 356)
- ✅ More balanced cluster sizes
- ✅ Faster computation
- ⚠️ Lower Davies-Bouldin (1.23 vs 0.99)

**Trade-off**: Slightly less tight clusters, but better overall separation and balance

### For 2-Class Task (Stressed/Unstressed):
**→ Use Hierarchical Clustering**

**Reasons:**
- ✅ Better Silhouette score (0.6462 vs 0.6034)
- ✅ Lower Davies-Bouldin (0.8254 vs 0.9577)
- ✅ **High agreement** with KMeans (ARI=0.78)
- ✅ No need to specify K in advance
- ⚠️ Lower Calinski-Harabasz (364 vs 393)

**Trade-off**: Slightly lower variance ratio, but tighter, better-defined clusters

---

## 🔬 Generated Visualizations

### 3-Class Comparison
📂 Location: `data/clustering_comparison/`

1. **clustering_methods_comparison.png**
   - Side-by-side PCA visualizations of both methods
   - Shows cluster assignments in 2D space

2. **clustering_metrics_chart.png**
   - Bar charts comparing all 3 quality metrics
   - Includes interpretation thresholds

3. **hierarchical_dendrogram.png**
   - Dendrogram showing hierarchical structure
   - Red line indicates 3-cluster cutoff

4. **cluster_size_comparison.png**
   - Side-by-side cluster size distributions
   - Highlights imbalance differences

5. **cluster_characteristics_heatmap.png**
   - Normalized feature values per cluster
   - Compares prosodic profiles between methods

### 2-Class Comparison
📂 Location: `data/clustering_comparison_binary/`

Same 5 visualizations for binary classification task

---

## 📁 Generated Data Files

### Comparison Metrics
- `clustering_metrics_comparison.csv` - Quality metrics table
- `clustering_agreement_metrics.csv` - Agreement scores (ARI, NMI, etc.)
- `cluster_characteristics.csv` - Mean features per cluster per method

### Reports
- `CLUSTERING_COMPARISON_REPORT.md` - Full detailed analysis
- `CLUSTERING_METHODS_SUMMARY.md` - This file

---

## 🚀 How to Use

### Run Clustering with Both Methods
```bash
# 3-class
python cluster_word_stress.py \
  --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --output-dir data/word_stress_clustered \
  --n-clusters 3 \
  --method both

# 2-class
python cluster_word_stress.py \
  --input-json data/prosody_arpabet_full/phoneme_features_arpabet.json \
  --output-dir data/word_stress_binary \
  --n-clusters 2 \
  --method both
```

### Run Comparison Analysis
```bash
# 3-class comparison
python compare_clustering_methods.py \
  --clustered-csv data/word_stress_clustered/word_stress_features.csv \
  --n-clusters 3 \
  --output-dir data/clustering_comparison

# 2-class comparison
python compare_clustering_methods.py \
  --clustered-csv data/word_stress_binary/word_stress_features.csv \
  --n-clusters 2 \
  --output-dir data/clustering_comparison_binary
```

---

## 📊 Results at a Glance

| Task | Best Method | Silhouette | Davies-Bouldin | Agreement |
|------|-------------|------------|----------------|-----------|
| **3-Class** | **KMeans** | 0.28 | 1.23 | LOW (0.16) |
| **2-Class** | **Hierarchical** | 0.65 | 0.83 | HIGH (0.78) |

### Interpretation

**3-Class task is harder:**
- Lower Silhouette scores (0.27-0.28 vs 0.60-0.65)
- Methods disagree more (ARI=0.16 vs 0.78)
- More subjective linguistic distinction between primary/secondary stress

**2-Class task is clearer:**
- High Silhouette scores (0.60-0.65)
- Methods largely agree (ARI=0.78)
- Stressed vs unstressed is acoustically distinct

---

## 🎓 Linguistic Insights

### Why 3-class is harder:
- Primary vs secondary stress is gradient, not categorical
- Context-dependent (sentence position, emphasis, speaker)
- Acoustically similar (both have higher pitch/duration than unstressed)

### Why 2-class is easier:
- Stressed vs unstressed is more binary
- Larger acoustic differences
- More consistent across speakers/contexts

### Cluster Characteristics

**Primary stress** (both methods agree):
- Highest pitch (440 Hz KMeans, similar in Hierarchical)
- Longest duration in some cases
- Highest prominence scores (>2.0)
- Rare (~4-5% of words)

**Secondary stress**:
- Medium pitch (207-270 Hz)
- Variable duration (longest vowels in KMeans: 0.33s)
- Medium prominence (0.16-0.60)
- Common (~40-75% depending on method)

**Unstressed**:
- Lowest pitch (87-254 Hz)
- Shorter duration (0.22-0.25s)
- Negative prominence scores (-0.66 to -1.06)
- Dominant class (~20-54%)

---

## 🔮 Future Work

1. **Try more clustering methods:**
   - DBSCAN (density-based)
   - Gaussian Mixture Models
   - Spectral clustering

2. **Feature engineering:**
   - Add formant features (F1, F2, F3)
   - Include speaking rate normalization
   - Add phonetic context features

3. **Semi-supervised learning:**
   - Use IPA stress markers as weak labels
   - Constrained clustering with phonological rules

4. **Deep learning:**
   - Autoencoder-based clustering
   - Transformer embeddings for prosody

---

**Generated**: January 13, 2026
**Dataset**: 1,051 words from LibriSpeech-style corpus
**Methods**: KMeans vs Hierarchical (Ward linkage)
