import argparse
import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".matplotlib_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


def load_payload(path: Path):
    data = json.loads(path.read_text())
    return data


def extract_rows(payload):
    rows = []
    for r in payload["results"]:
        diag = r.get("diagnostics", {})
        support = diag.get("support_counts", {})
        support_total = sum(int(v) for v in support.values()) if support else 0
        rows.append(
            {
                "kernel": r["name"],
                "cv": float(r["cv_score"]),
                "train": float(r["train_acc"]),
                "val": float(r["val_acc"]),
                "fit_s": float(r.get("fit_time_sec", 0.0)),
                "misclassified": int(diag.get("misclassified_count", 0)),
                "boundary": int(diag.get("boundary_point_count", 0)),
                "support_total": support_total,
                "support_per_class_mean": float(np.mean(list(support.values()))) if support else 0.0,
            }
        )
    return rows


def make_plots(payload, output_png, output_md):
    rows = extract_rows(payload)
    labels = [r["kernel"] for r in rows]
    cv = np.array([r["cv"] for r in rows])
    train = np.array([r["train"] for r in rows])
    val = np.array([r["val"] for r in rows])
    fit = np.array([r["fit_s"] for r in rows])
    mis = np.array([r["misclassified"] for r in rows])
    boundary = np.array([r["boundary"] for r in rows])
    support = np.array([r["support_total"] for r in rows])
    support_mean = np.array([r["support_per_class_mean"] for r in rows])

    gap = train - val
    boundary_share = boundary / val.size if len(rows) else np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    x = np.arange(len(labels))
    width = 0.24

    # Accuracy panel
    axes[0, 0].bar(x - width, cv, width, label="CV score", color="#4e79a7")
    axes[0, 0].bar(x, train, width, label="Train accuracy", color="#f28e2b")
    axes[0, 0].bar(x + width, val, width, label="Validation accuracy", color="#59a14f")
    axes[0, 0].set_title("Accuracy by kernel")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=12)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    for i, (cv_v, tr_v, va_v) in enumerate(zip(cv, train, val)):
        axes[0, 0].text(i - width, cv_v + 0.02, f"{cv_v:.2f}", ha="center", va="bottom", fontsize=7)
        axes[0, 0].text(i, tr_v + 0.02, f"{tr_v:.2f}", ha="center", va="bottom", fontsize=7)
        axes[0, 0].text(i + width, va_v + 0.02, f"{va_v:.2f}", ha="center", va="bottom", fontsize=7)

    # Overfit gap panel
    axes[0, 1].bar(x, gap, color="#e15759")
    axes[0, 1].set_title("Train - Validation gap")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=12)
    axes[0, 1].set_ylabel("Gap")
    for i, g in enumerate(gap):
        axes[0, 1].text(i, g + 0.01, f"{g:.2f}", ha="center", va="bottom", fontsize=8)

    # Boundary and misclassification panel
    axes[1, 0].bar(x - width / 2, mis, width, label="Misclassified train", color="#9c755f")
    axes[1, 0].bar(x + width / 2, boundary, width, label="Boundary points", color="#76b7b2")
    axes[1, 0].set_title("Complexity signals")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=12)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    # Support vectors panel
    axes[1, 1].bar(x - width / 2, support, width, label="Total support vectors", color="#b07aa1")
    axes[1, 1].bar(x + width / 2, support_mean, width, label="Mean support / class", color="#ff9da7")
    axes[1, 1].set_title("Support-vector usage")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=12)
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    fig.suptitle("20 Newsgroups SVC Comparison (OvR)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    # Write short explanation markdown
    best_val = max(rows, key=lambda r: r["val"])
    best_cv = max(rows, key=lambda r: r["cv"])
    worst_val = min(rows, key=lambda r: r["val"])
    best_gap = min(rows, key=lambda r: r["train"] - r["val"])
    largest_train = max(rows, key=lambda r: r["train"])

    txt = [
        "# 20 Newsgroups SVC Result Summary",
        "",
        f"Dataset config: {payload.get('config', {})}",
        "",
        f"- Best validation accuracy: **{best_val['kernel']}** ({best_val['val']:.3f})",
        f"- Best CV score: **{best_cv['kernel']}** ({best_cv['cv']:.3f})",
        f"- Smallest train-validation gap: **{best_gap['kernel']}** ({best_gap['train'] - best_gap['val']:.3f})",
        f"- Lowest validation accuracy: **{worst_val['kernel']}** ({worst_val['val']:.3f})",
        "",
        "## Numerical summary",
        "",
    ]
    txt.append("| kernel | cv | train | val | train-val | misclassified | boundary | total SV | mean SV/class |")
    txt.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        txt.append(
            "| {k} | {cv:.3f} | {tr:.3f} | {va:.3f} | {gap:.3f} | {mis} | {bnd} | {sv} | {svm:.1f} |".format(
                k=r["kernel"],
                cv=r["cv"],
                tr=r["train"],
                va=r["val"],
                gap=(r["train"] - r["val"]),
                mis=r["misclassified"],
                bnd=r["boundary"],
                sv=r["support_total"],
                svm=r["support_per_class_mean"],
            )
        )
    txt.extend(
        [
            "",
            "### Interpretation",
            "- Train accuracy being much larger than validation accuracy suggests overfitting.",
            "- Larger support-vector counts usually indicate more complex decision boundaries for that OvR set.",
            "- Boundary points are samples with small margin; many of them imply unstable class separation in this feature setting.",
            "- In this experiment, linear models tend to generalize better than Matern and are competitive with RBF on validation.",
            "",
        ]
    )
    output_md.write_text("\n".join(txt))
    return output_png, output_md


def main():
    parser = argparse.ArgumentParser(description="Visualize 20 Newsgroups OvR SVC JSON results")
    parser.add_argument("input_json", nargs="?", default="smoke_results.json")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--plot", default="results_overview.png")
    parser.add_argument("--report", default="results_explanation.md")
    args = parser.parse_args()

    if args.run_dir:
        base = Path(args.run_dir).resolve()
        if args.run_name:
            base = base / args.run_name
        base.mkdir(parents=True, exist_ok=True)
        payload_path = base / args.input_json
        plot_path = base / args.plot
        report_path = base / args.report
    else:
        payload_path = Path(args.input_json)
        plot_path = Path(args.plot)
        report_path = Path(args.report)

    payload = load_payload(payload_path)
    make_plots(payload, plot_path, report_path)
    print(f"Wrote {plot_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
