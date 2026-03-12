import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import Matern


def matern_kernel_factory(length_scale: float = 1.0, nu: float = 1.5):
    kernel = Matern(length_scale=length_scale, nu=nu, length_scale_bounds="fixed")

    def matern_kernel(X, Y):
        X = np.asarray(X.toarray()) if sp.issparse(X) else np.asarray(X)
        Y = np.asarray(Y.toarray()) if sp.issparse(Y) else np.asarray(Y)
        return kernel(X, Y)

    matern_kernel.__name__ = f"matern_ls{length_scale}_nu{nu}"
    return matern_kernel


def pick_stratified_subset(X, y, size: int | None, seed: int):
    if size is None or size >= len(y):
        return X, y
    idx = np.arange(len(y))
    selected, _, _, _ = train_test_split(
        idx,
        idx,
        train_size=size / len(y),
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    return X[selected], y[selected]


def build_experiment_data(
    data_home: str,
    max_train_samples: int | None,
    seed: int,
    val_size: float,
):
    data = fetch_20newsgroups_vectorized(subset="train", data_home=data_home)
    X, y = data.data, data.target
    target_names = data.target_names

    X_small, y_small = pick_stratified_subset(X, y, max_train_samples, seed)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_small,
        y_small,
        test_size=val_size,
        random_state=seed,
        stratify=y_small,
    )
    return X_fit, X_val, y_fit, y_val, target_names


def run_grid_search(
    name: str,
    base_estimator,
    grid: Dict,
    X_train,
    y_train,
    X_val,
    y_val,
    cv: int,
    n_jobs: int,
):
    gs = GridSearchCV(
        base_estimator,
        param_grid=grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="accuracy",
        verbose=2,
        refit=True,
    )
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    best = gs.best_estimator_
    train_pred = best.predict(X_train)
    val_pred = best.predict(X_val)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    return {
        "name": name,
        "best_params": gs.best_params_,
        "cv_score": gs.best_score_,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "fit_time_sec": fit_time,
        "train_report": classification_report(y_train, train_pred, output_dict=True, zero_division=0),
        "val_report": classification_report(y_val, val_pred, output_dict=True, zero_division=0),
        "model": best,
    }


def inspect_ovr_svc(model: OneVsRestClassifier, X_train, y_train, target_names, boundary_quantile=0.01, max_boundary=20):
    train_pred = model.predict(X_train)
    misclassified = np.flatnonzero(train_pred != y_train)

    support = {}
    decision = model.decision_function(X_train)
    if decision.ndim == 1:
        decision = decision[:, None]

    closest_to_boundary = np.argsort(np.abs(decision).min(axis=1))
    boundary_indices = set(closest_to_boundary[: max_boundary])

    per_class_boundary: Dict[int, List[int]] = {}
    per_class_support: Dict[int, List[int]] = {}
    for cls_i, est in enumerate(model.estimators_):
        cls = int(model.classes_[cls_i])
        support_idx = [int(i) for i in est.support_]
        per_class_support[cls] = support_idx

        margins = est.decision_function(X_train)
        thresh = np.quantile(np.abs(margins), boundary_quantile)
        near_zero = set(np.flatnonzero(np.abs(margins) <= thresh).tolist())
        if not near_zero:
            near_zero.add(int(np.argmin(np.abs(margins))))
        per_class_boundary[cls] = sorted(near_zero)
        boundary_indices.update(near_zero)

    return {
        "misclassified_count": int(len(misclassified)),
        "misclassified_indices": [int(i) for i in misclassified[:200]],
        "support_counts": {int(k): len(v) for k, v in per_class_support.items()},
        "support_vectors_per_class_head": {
            target_names[k] if k < len(target_names) else str(k): v[:max_boundary]
            for k, v in per_class_support.items()
        },
        "boundary_point_count": int(len(boundary_indices)),
        "boundary_indices": [int(i) for i in sorted(boundary_indices)],
        "boundary_points_per_class_head": {
            target_names[k] if k < len(target_names) else str(k): v[:max_boundary]
            for k, v in per_class_boundary.items()
        },
    }


@dataclass
class Config:
    seed: int
    max_train_samples: int | None
    max_matern_train_samples: int | None
    svd_components: int
    cv: int
    n_jobs: int
    boundary_quantile: float
    val_size: float


def run_all(args):
    data_home = Path(args.data_home).resolve()
    config = Config(
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_matern_train_samples=args.max_matern_train_samples,
        svd_components=args.svd_components,
        cv=args.cv,
        n_jobs=args.n_jobs,
        boundary_quantile=args.boundary_quantile,
        val_size=args.val_size,
    )

    X_fit, X_val, y_fit, y_val, target_names = build_experiment_data(
        data_home=str(data_home),
        max_train_samples=config.max_train_samples,
        seed=config.seed,
        val_size=config.val_size,
    )

    results = []

    linear = OneVsRestClassifier(
        SVC(kernel="linear", class_weight="balanced", random_state=config.seed, max_iter=-1)
    )
    linear_grid = {"estimator__C": [0.1, 1.0, 3.0, 10.0]}
    linear_out = run_grid_search(
        "linear",
        linear,
        linear_grid,
        X_fit,
        y_fit,
        X_val,
        y_val,
        cv=config.cv,
        n_jobs=config.n_jobs,
    )
    linear_out["diagnostics"] = inspect_ovr_svc(
        linear_out["model"],
        X_fit,
        y_fit,
        target_names,
        boundary_quantile=config.boundary_quantile,
    )
    results.append({k: v for k, v in linear_out.items() if k != "model"})

    rbf = OneVsRestClassifier(
        SVC(kernel="rbf", class_weight="balanced", random_state=config.seed)
    )
    rbf_grid = {
        "estimator__C": [1.0, 3.0, 10.0],
        "estimator__gamma": ["scale", 1e-3, 1e-2, 1e-1],
    }
    rbf_out = run_grid_search(
        "rbf",
        rbf,
        rbf_grid,
        X_fit,
        y_fit,
        X_val,
        y_val,
        cv=config.cv,
        n_jobs=config.n_jobs,
    )
    rbf_out["diagnostics"] = inspect_ovr_svc(
        rbf_out["model"],
        X_fit,
        y_fit,
        target_names,
        boundary_quantile=config.boundary_quantile,
    )
    results.append({k: v for k, v in rbf_out.items() if k != "model"})

    if config.max_matern_train_samples and len(y_fit) > config.max_matern_train_samples:
        idx = np.arange(len(y_fit))
        idx_m, _ = train_test_split(
            idx,
            train_size=config.max_matern_train_samples / len(y_fit),
            random_state=config.seed,
            stratify=y_fit,
            shuffle=True,
        )
        X_fit_for_m = X_fit[idx_m]
        y_fit_for_m = y_fit[idx_m]
    else:
        X_fit_for_m = X_fit
        y_fit_for_m = y_fit

    svd = TruncatedSVD(n_components=config.svd_components, random_state=config.seed)
    X_fit_m = svd.fit_transform(X_fit_for_m)
    X_val_m = svd.transform(X_val)
    matern = OneVsRestClassifier(
        SVC(class_weight="balanced", random_state=config.seed, cache_size=500)
    )
    matern_grid = {
        "estimator__C": [1.0, 3.0, 10.0],
        "estimator__kernel": [
            matern_kernel_factory(0.5, 0.5),
            matern_kernel_factory(1.0, 0.5),
            matern_kernel_factory(2.0, 0.5),
            matern_kernel_factory(1.0, 1.5),
            matern_kernel_factory(2.0, 1.5),
        ],
    }
    matern_out = run_grid_search(
        "matern",
        matern,
        matern_grid,
        X_fit_m,
        y_fit_for_m,
        X_val_m,
        y_val,
        cv=min(config.cv, 3),
        n_jobs=1,
    )
    matern_out["diagnostics"] = inspect_ovr_svc(
        matern_out["model"],
        X_fit_m,
        y_fit_for_m,
        target_names,
        boundary_quantile=config.boundary_quantile,
    )
    results.append({k: v for k, v in matern_out.items() if k != "model"})

    return {
        "config": asdict(config),
        "results": results,
        "target_names": target_names,
        "model_cache": {
            "linear": linear_out["model"],
            "rbf": rbf_out["model"],
            "matern": matern_out["model"],
        },
    }


def _json_safe_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    return value


def main():
    parser = argparse.ArgumentParser(
        description="20 Newsgroups OvR SVC comparison (linear, RBF, Matern)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-home", type=str, default=".")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-matern-train-samples", type=int, default=2500)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--svd-components", type=int, default=120)
    parser.add_argument("--boundary-quantile", type=float, default=0.01)
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Filename or path for results JSON (inside run directory if relative)",
    )
    args = parser.parse_args()

    payload = run_all(args)

    print("Comparative summary")
    for r in payload["results"]:
        print(
            f"{r['name']}: cv={r['cv_score']:.4f}, "
            f"train={r['train_acc']:.4f}, val={r['val_acc']:.4f}, "
            f"best={r['best_params']}"
        )
        print(
            f"  misclassified train={r['diagnostics']['misclassified_count']}, "
            f"boundary={r['diagnostics']['boundary_point_count']}, "
            f"support vectors total classes={len(r['diagnostics']['support_counts'])}"
        )

    run_dir = Path(args.run_dir).resolve()
    if args.run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    else:
        run_name = args.run_name
    run_dir = run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_json is None:
        out_path = run_dir / "results.json"
    else:
        candidate = Path(args.summary_json)
        out_path = candidate if candidate.is_absolute() else run_dir / candidate

    serializable_results = []
    for r in payload["results"]:
        safe_best_params = {k: _json_safe_value(v) for k, v in r["best_params"].items()}
        serializable_results.append(
            {
                "name": r["name"],
                "best_params": safe_best_params,
                "cv_score": float(r["cv_score"]),
                "train_acc": float(r["train_acc"]),
                "val_acc": float(r["val_acc"]),
                "fit_time_sec": float(r["fit_time_sec"]),
                "diagnostics": r["diagnostics"],
            }
        )
    serializable = {
        "config": payload["config"],
        "results": serializable_results,
    }
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
