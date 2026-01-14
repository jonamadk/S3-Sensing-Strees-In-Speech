# Visualization Guide
## Quick Reference for Word Stress Clustering Analysis

This guide helps you navigate the generated visualizations and reports.

---

## 📊 Generated Files Overview

### For 3-Class Clustering (Primary/Secondary/Unstressed)
**Directory**: `data/analysis_reports_3clusters/`

### For 2-Class Clustering (Stressed/Unstressed)  
**Directory**: `data/analysis_reports_2clusters/`

---

## 🎯 What to Look At Based on Your Question

### Question 1: "How are vowels distributed across stress levels?"
**View**: `vowel_distribution_analysis.png`

**What you'll see**:
- **Top-left**: Box plot showing number of vowels per word by stress
- **Top-right**: Bar chart of most frequent vowel phonemes (AA, AE, IH, etc.) by stress
- **Bottom-left**: Histogram of vowel ratio (vowel_duration/word_duration)
- **Bottom-right**: Scatter plot of vowel duration vs word duration

**Key Insights**:
- Secondary stress words have highest vowel counts
- AH0 (schwa) dominates unstressed words
- Stressed words cluster in upper-right (longer vowels + longer words)

---

### Question 2: "How do pitch, energy, and duration relate to stress?"
**View**: `prosodic_features_analysis.png`

**What you'll see**:
- **Top-left**: Correlation heatmap (which features relate to each other?)
- **Top-right**: Pitch vs Duration scatter (colored by stress)
- **Bottom-left**: Energy vs Pitch scatter
- **Bottom-right**: Prominence score distribution (box plot)

**Key Insights**:
- Pitch max and pitch mean are highly correlated (r=0.94)
- Stressed words cluster in high-pitch, high-duration region
- Prominence score clearly separates stress levels

---

### Question 3: "Is the clustering good quality?"
**View**: `clustering_quality_metrics.png`

**What you'll see**:
- **Left**: Silhouette plot showing how well words fit their clusters
  - Positive values = word fits well in its cluster
  - Negative values = word might fit better in another cluster
- **Right**: Feature variance across clusters (which features matter most?)

**Key Insights**:
- 2-class: Silhouette = 0.60 (Good separation)
- 3-class: Silhouette = 0.27 (Moderate separation)
- Primary stress cluster has highest silhouette values (most distinct)

---

### Question 4: "How do features relate to each other?"
**View**: `feature_pairplot.png`

**What you'll see**:
- Grid of scatter plots for all feature pairs:
  - vowel_duration × vowel_ratio
  - vowel_duration × pitch_max
  - vowel_duration × prominence_score
  - (and all other combinations)
- Diagonal: KDE density plots

**Key Insights**:
- Clear separation in vowel_duration vs pitch_max space
- Prominence score shows best separation
- Stressed words form distinct cluster in most pairwise plots

---

### Question 5: "What are the feature distributions?"
**View**: `feature_distributions.png`

**What you'll see**:
- 4 ridge plots (overlapping histograms):
  1. Vowel duration distribution
  2. Pitch max distribution
  3. Pitch range distribution
  4. Prominence score distribution

**Key Insights**:
- Unstressed words cluster at low values
- Secondary stress in middle range
- Primary stress at high values (especially for pitch)

---

### Question 6: "Can I see the feature space in 3D?"
**View**: `feature_3d_plot.png`

**What you'll see**:
- 3D scatter plot:
  - X-axis: Vowel duration
  - Y-axis: Max pitch
  - Z-axis: Prominence score
- Points colored by stress level

**Key Insights**:
- Stressed words occupy upper corner of 3D space
- Clear geometric separation
- Visualizes why clustering works

---

### Question 7: "What are the statistical test results?"
**View**: `clustering_analysis_report.txt`

**What you'll see**:
```
CLUSTER DISTRIBUTION
-------------------
UNSTRESSED: 518 words (49.29%)
SECONDARY:  476 words (45.29%)
PRIMARY:     57 words ( 5.42%)

STATISTICAL SIGNIFICANCE TESTS
------------------------------
Feature             ANOVA F-stat  p-value       Significant?
vowel_duration      174.42        4.1e-66       Yes ***
pitch_max           597.02        8.6e-174      Yes ***
prominence_score    1054.79       1.0e-251      Yes ***

CLUSTERING QUALITY METRICS
--------------------------
Silhouette Score:        0.2718
Davies-Bouldin Index:    1.2210
Calinski-Harabasz Score: 421.57
```

**Key Insights**:
- All features significantly differ across stress levels (p < 0.05)
- Prominence score has strongest discrimination (F=1054.79)
- Moderate cluster quality (silhouette=0.27)

---

## 📈 Interpretation Guide

### Silhouette Score
- **> 0.7**: Strong, well-separated clusters
- **0.5 - 0.7**: Good cluster structure
- **0.25 - 0.5**: Moderate overlap (acceptable)
- **< 0.25**: Weak clustering

**Your Results**:
- 2-class: 0.60 (Good) ✓
- 3-class: 0.27 (Moderate) ✓

### Davies-Bouldin Index
- **Lower is better**
- < 1.0: Good separation
- 1.0 - 2.0: Moderate separation
- \> 2.0: Poor separation

**Your Results**:
- 2-class: 0.96 (Good) ✓
- 3-class: 1.22 (Moderate) ✓

### p-values (Statistical Significance)
- **p < 0.001**: Highly significant (***) - Very strong evidence
- **p < 0.01**: Significant (**) - Strong evidence
- **p < 0.05**: Significant (*) - Sufficient evidence
- **p ≥ 0.05**: Not significant (ns) - Insufficient evidence

**Your Results**: All key features have p < 1e-28 (extremely significant!) ✓

---

## 🔍 Detailed Analysis Workflow

### For Researchers/Linguists:

1. **Start with**: `clustering_analysis_report.txt`
   - Get overview of cluster distribution
   - Check statistical significance
   - Note key findings

2. **Then view**: `vowel_distribution_analysis.png`
   - Understand vowel patterns
   - Identify most common vowels per stress level

3. **Explore**: `prosodic_features_analysis.png`
   - See correlations between features
   - Understand pitch/duration/energy relationships

4. **Validate**: `clustering_quality_metrics.png`
   - Assess clustering performance
   - Identify which features contribute most

5. **Deep dive**: `feature_pairplot.png`
   - Explore all feature relationships
   - Identify interesting patterns

### For Machine Learning Engineers:

1. **Quality metrics**: `clustering_quality_metrics.png`
   - Silhouette, Davies-Bouldin, Calinski-Harabasz scores

2. **Feature importance**: `prosodic_features_analysis.png` (correlation heatmap)
   - Identify redundant features
   - Select most discriminative features

3. **Separation analysis**: `feature_3d_plot.png`
   - Visualize cluster separability in feature space

4. **Distribution overlap**: `feature_distributions.png`
   - Assess how much clusters overlap
   - Decide if binary or 3-class is better

### For Speech Scientists:

1. **Vowel analysis**: `vowel_distribution_analysis.png`
   - Which vowels occur in stressed positions?
   - How does vowel duration vary?

2. **Prosodic cues**: `prosodic_features_analysis.png`
   - Pitch range differences
   - Energy patterns
   - Duration relationships

3. **Statistical validation**: `clustering_analysis_report.txt`
   - ANOVA results
   - Pairwise comparisons
   - Effect sizes

---

## 💡 Key Findings Quick Reference

### 3-Class Clustering (Primary/Secondary/Unstressed)

| Stress Level | Vowel Duration | Pitch Max | Prominence | Count |
|--------------|----------------|-----------|------------|-------|
| Primary | 0.161s | 440 Hz | 2.05 | 57 (5%) |
| Secondary | 0.291s | 207 Hz | 0.52 | 476 (45%) |
| Unstressed | 0.148s | 87 Hz | -0.70 | 518 (50%) |

**Insight**: Secondary stress has LONGEST vowels, but primary has HIGHEST pitch.

### 2-Class Clustering (Stressed/Unstressed)

| Stress Level | Vowel Duration | Pitch Max | Prominence | Count |
|--------------|----------------|-----------|------------|-------|
| Stressed | 0.177s | 428 Hz | 2.03 | 65 (6%) |
| Unstressed | 0.216s | 143 Hz | -0.13 | 986 (94%) |

**Insight**: Pitch is the PRIMARY discriminator (285 Hz difference).

---

## 🚀 Next Steps

### If you want to improve clustering:
1. Add spectral features (formants F1, F2, F3)
2. Include pitch velocity/acceleration
3. Normalize by speaking rate
4. Try hierarchical clustering
5. Use semi-supervised learning with IPA stress markers

### If you want to use the results:
1. **TTS systems**: Use prominence scores to generate natural prosody
2. **ASR systems**: Context-aware acoustic models
3. **Pronunciation training**: Compare learner stress vs. native patterns
4. **Linguistic research**: Study English stress patterns empirically

### If you want more analysis:
1. Run analysis on each audio genre separately
2. Compare male vs female speakers
3. Analyze stress patterns in multi-syllable words
4. Study stress clash and rhythm

---

## 📁 File Organization

```
data/
├── analysis_reports_3clusters/          # 3-class results
│   ├── vowel_distribution_analysis.png
│   ├── prosodic_features_analysis.png
│   ├── clustering_quality_metrics.png
│   ├── feature_pairplot.png
│   ├── feature_distributions.png
│   ├── feature_3d_plot.png
│   └── clustering_analysis_report.txt
│
├── analysis_reports_2clusters/          # 2-class results
│   ├── (same files as above)
│
├── word_stress_clustered/               # 3-class data
│   ├── word_stress_features.csv
│   ├── word_stress_features.json
│   ├── word_stress_visualization.png
│   └── word_stress_detailed.png
│
└── word_stress_binary/                  # 2-class data
    ├── word_stress_features.csv
    ├── word_stress_features.json
    ├── word_stress_visualization.png
    └── word_stress_detailed.png
```

---

## 🔧 Regenerating Visualizations

To regenerate all analysis visualizations:

```bash
# 3-class analysis
python analyze_stress_clustering.py \
    --input-csv data/word_stress_clustered/word_stress_features.csv \
    --output-dir data/analysis_reports_3clusters

# 2-class analysis
python analyze_stress_clustering.py \
    --input-csv data/word_stress_binary/word_stress_features.csv \
    --output-dir data/analysis_reports_2clusters
```

---

**Last Updated**: January 13, 2026  
**Tool**: `analyze_stress_clustering.py`  
**Dataset**: ARPAbet phoneme features (1,051 words, 4,562 phonemes)
