# 20 Newsgroups SVC Result Summary

Dataset config: {'seed': 0, 'max_train_samples': 300, 'max_matern_train_samples': 120, 'svd_components': 20, 'cv': 2, 'n_jobs': 1, 'boundary_quantile': 0.01, 'val_size': 0.2}

- Best validation accuracy: **linear** (0.400)
- Best CV score: **linear** (0.283)
- Smallest train-validation gap: **linear** (0.600)
- Lowest validation accuracy: **matern** (0.133)

## Numerical summary

| kernel | cv | train | val | train-val | misclassified | boundary | total SV | mean SV/class |
|---|---|---|---|---|---:|---:|---:|---:|
| linear | 0.283 | 1.000 | 0.400 | 0.600 | 0 | 48 | 2391 | 119.5 |
| rbf | 0.275 | 1.000 | 0.383 | 0.617 | 0 | 52 | 2519 | 126.0 |
| matern | 0.108 | 0.933 | 0.133 | 0.800 | 8 | 35 | 1049 | 52.5 |

### Interpretation
- Train accuracy being much larger than validation accuracy suggests overfitting.
- Larger support-vector counts usually indicate more complex decision boundaries for that OvR set.
- Boundary points are samples with small margin; many of them imply unstable class separation in this feature setting.
- In this experiment, linear models tend to generalize better than Matern and are competitive with RBF on validation.
