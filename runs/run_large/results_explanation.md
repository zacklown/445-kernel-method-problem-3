# 20 Newsgroups SVC Result Summary

Dataset config: {'seed': 0, 'max_train_samples': 2000, 'max_matern_train_samples': 1000, 'svd_components': 80, 'cv': 3, 'n_jobs': 1, 'boundary_quantile': 0.01, 'val_size': 0.2}

- Best validation accuracy: **rbf** (0.772)
- Best CV score: **linear** (0.717)
- Smallest train-validation gap: **rbf** (0.228)
- Lowest validation accuracy: **matern** (0.407)

## Numerical summary

| kernel | cv | train | val | train-val | misclassified | boundary | total SV | mean SV/class |
|---|---|---|---|---|---:|---:|---:|---:|
| linear | 0.717 | 1.000 | 0.765 | 0.235 | 0 | 291 | 9137 | 456.9 |
| rbf | 0.711 | 1.000 | 0.772 | 0.228 | 0 | 280 | 10420 | 521.0 |
| matern | 0.444 | 1.000 | 0.407 | 0.593 | 0 | 187 | 7304 | 365.2 |

### Interpretation
- Train accuracy being much larger than validation accuracy suggests overfitting.
- Larger support-vector counts usually indicate more complex decision boundaries for that OvR set.
- Boundary points are samples with small margin; many of them imply unstable class separation in this feature setting.
- In this experiment, linear models tend to generalize better than Matern and are competitive with RBF on validation.
