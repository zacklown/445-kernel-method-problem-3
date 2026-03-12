# 20 Newsgroups OvR SVC experiments

This repository contains scripts to run and visualize multiclass text classification experiments on the 20 Newsgroups dataset using one-vs-rest SVMs with three kernels:

- linear
- RBF
- Matern (custom callable kernel)

## What's included

- `analyze_20newsgroups_svc.py`
  - Loads vectorized 20 Newsgroups data
  - Trains OvR models for each kernel
  - Tunes hyperparameters with `GridSearchCV`
  - Evaluates train/validation performance
  - Reports support vectors, misclassified train samples, and near-boundary points
  - Writes a run JSON report
- `visualize_20ng_results.py`
  - Reads a run JSON
  - Generates charts and a short explanation artifact

## Environment setup

```bash
python -m pip install scikit-learn numpy scipy matplotlib
```

The code also uses `scipy.sparse`, which comes with SciPy.

## Data

The code calls `fetch_20newsgroups_vectorized` and can also use existing local vectorized files if already present in your workspace.

- Default data location is the current directory (`--data-home .`)
- You can point this to the directory containing the local `20newsgroups` cache

## Recommended run patterns

### Quick smoke run

Use a smaller subset for fast turnaround while validating the pipeline:

```bash
python analyze_20newsgroups_svc.py \
  --max-train-samples 400 \
  --max-matern-train-samples 200 \
  --svd-components 40 \
  --cv 2 \
  --val-size 0.2 \
  --seed 0 \
  --n-jobs 1 \
  --run-dir runs \
  --run-name smoke
```

### Larger run

Use the larger configuration you've been using:

```bash
python analyze_20newsgroups_svc.py \
  --max-train-samples 2000 \
  --max-matern-train-samples 1000 \
  --svd-components 80 \
  --cv 3 \
  --val-size 0.2 \
  --seed 0 \
  --n-jobs 1 \
  --run-dir runs \
  --run-name run_large
```

### Visualize a run

```bash
python visualize_20ng_results.py --run-dir runs --run-name run_large
```

If you omit `--run-name`, the script uses the default `run_*` folder name convention.

## Script outputs

Each analysis run creates a directory under `runs/<run_name>/` with:

- `results.json`: numerical summary
- `results_explanation.md`: generated interpretation text
- `results_overview.png`: comparison plots

For a run name override, all three files use that folder.

## Useful arguments (analysis script)

From `analyze_20newsgroups_svc.py`:

- `--data-home`: dataset cache path
- `--max-train-samples`: max training docs for linear/RBF
- `--max-matern-train-samples`: max training docs for Matern (subsampled for performance)
- `--svd-components`: SVD dimensions before Matern SVC
- `--cv`: cross-validation folds
- `--val-size`: validation split fraction
- `--seed`: RNG seed
- `--n-jobs`: parallel jobs for `GridSearchCV` (set conservatively for OS constraints)
- `--run-dir`, `--run-name`: control run output location
- `--summary-json`: custom JSON output filename inside the run directory

## Notes

- The pipeline is CPU-bound (scikit-learn `SVC` implementation used here).
- The Matern configuration is intentionally reduced (subsampling + SVD) to keep runtime manageable.
- To reproduce large-run settings exactly, keep the same seed and split/grid settings recorded in the output JSON.

## Git hygiene

This repo includes a `.gitignore` for:

- Python caches and virtual envs
- Large serialized artifacts (`*.pkl`, `*.pkz`, `data/`)
- Full run outputs (`runs/`)
- Plot images (`*.png`)

If you want to track a specific output file, add it explicitly with `git add -f` and an un-ignore override.
