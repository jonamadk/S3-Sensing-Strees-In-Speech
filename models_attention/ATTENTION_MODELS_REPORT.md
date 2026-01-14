# Attention-Based Models for Word Stress Prediction

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Transformer | 0.9919 | 0.9921 | 0.9919 | 0.9920 |
| Feature Attention | 0.9881 | 0.9881 | 0.9881 | 0.9881 |
| Multi-Head Attention | 0.9782 | 0.9786 | 0.9782 | 0.9781 |

## Detailed Results

### Feature Attention

```
              precision    recall  f1-score   support

  Unstressed       0.98      0.99      0.99       612
   Secondary       0.99      0.99      0.99      1388
     Primary       0.96      0.94      0.95       107

    accuracy                           0.99      2107
   macro avg       0.98      0.98      0.98      2107
weighted avg       0.99      0.99      0.99      2107
```

### Multi-Head Attention

```
              precision    recall  f1-score   support

  Unstressed       1.00      0.94      0.97       612
   Secondary       0.97      0.99      0.98      1388
     Primary       0.94      0.96      0.95       107

    accuracy                           0.98      2107
   macro avg       0.97      0.97      0.97      2107
weighted avg       0.98      0.98      0.98      2107
```

### Transformer

```
              precision    recall  f1-score   support

  Unstressed       1.00      0.99      0.99       612
   Secondary       0.99      0.99      0.99      1388
     Primary       0.94      0.97      0.95       107

    accuracy                           0.99      2107
   macro avg       0.98      0.99      0.98      2107
weighted avg       0.99      0.99      0.99      2107
```


## Training Summary

- **Architecture**: Attention-based neural networks
- **Attention Types**: Feature-wise, Multi-head, Transformer
- **Framework**: PyTorch
- **Early Stopping**: Patience = 20 epochs
- **Optimization**: Adam optimizer

## Key Findings

- **Best Model**: Transformer (F1: 0.9920)
- **Classes**: Unstressed, Secondary, Primary
- **Attention Mechanism**: Learns feature importance automatically

