# xgboost / catboost approach

## Training

1. MFCC was calculated and zero-padded to size: (?, 100, 13)
2. For training feature matrix was flattened: (?, 100 * 13)
3. ... Training process ...
4. Accuracy on test: 45%
