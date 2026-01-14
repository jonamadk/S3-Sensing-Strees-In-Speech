# Attention Models Generalization Analysis

## Summary

This report evaluates model generalization using:
- **5-Fold Cross-Validation**: Tests performance across different data splits
- **Bootstrap Validation**: Tests stability with repeated random sampling
- **Overfitting Detection**: Analyzes train-validation gap

## 5-Fold Cross-Validation Results

| Model | Train F1 (Mean±SD) | Val F1 (Mean±SD) | Train-Val Gap | Status |
|-------|-------------------|------------------|---------------|--------|
| Feature Attention | 0.9940±0.0018 | 0.9922±0.0030 | 0.0018 | ✅ Excellent |
| Multi-Head Attention | 0.9824±0.0025 | 0.9865±0.0027 | -0.0041 | ✅ Excellent |
| Transformer | 0.9922±0.0039 | 0.9926±0.0020 | -0.0004 | ✅ Excellent |

### Interpretation:
- **Train-Val Gap < 1%**: Excellent generalization, no overfitting
- **Train-Val Gap 1-2%**: Good generalization, acceptable
- **Train-Val Gap > 2%**: Potential overfitting, needs regularization

## Bootstrap Validation Results (10 iterations)

| Model | Mean F1 | Std Dev | 95% CI | Stability |
|-------|---------|---------|--------|----------|
| Feature Attention | 0.9859 | 0.0032 | [0.9811, 0.9904] | 🟢 Highly Stable |
| Multi-Head Attention | 0.9761 | 0.0043 | [0.9685, 0.9808] | 🟢 Highly Stable |
| Transformer | 0.9897 | 0.0019 | [0.9870, 0.9920] | 🟢 Highly Stable |

### Interpretation:
- **Low Std Dev**: Model performs consistently across different samples
- **Tight 95% CI**: Predictions are reliable and stable
- **High variance**: Model is sensitive to training data composition

## Key Findings

1. **Best Generalizing Model**: Transformer (CV F1: 0.9926)
2. **Most Stable Model**: Transformer (Bootstrap Std: 0.0019)
3. **Recommended for Production**: Transformer

## Generalization Techniques Applied

✅ Early stopping (patience=20 epochs)
✅ Dropout regularization (0.3)
✅ Batch normalization
✅ Train-validation split (80:20)
✅ Cross-validation testing
✅ Bootstrap validation

## Recommendations

✅ **All models show excellent generalization**
- Train-validation gaps are minimal (<1%)
- Models are ready for production deployment
