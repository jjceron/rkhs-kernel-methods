"""
Experiment 2: Failure of Linear Models on Nonlinear Data.

Demonstrates that linear SVMs fail on:
  - Concentric circles (make_circles)
  - Two moons (make_moons)

Generates publication-quality figures showing the dramatic
limitation of linear separability.

Outputs:
  figures/decision_boundaries/02_linear_failure_circles.png
  figures/decision_boundaries/02_linear_failure_moons.png
  figures/decision_boundaries/02_linear_failure_combined.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rkhs_kernel_methods.datasets import (
    make_nonlinear_datasets,
)
from rkhs_kernel_methods.models import train_linear_svm
from rkhs_kernel_methods.visualization import (
    set_style,
    plot_decision_boundary,
    save_figure,
)
from rkhs_kernel_methods.evaluation import evaluate_model
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


def main() -> None:
    set_style()
    set_random_seed(42)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE, "figures", "decision_boundaries")
    ensure_dir(out_dir)

    print("=" * 60)
    print("EXPERIMENT 2: Failure of Linear Models")
    print("=" * 60)

    datasets = make_nonlinear_datasets(n_samples=300, random_state=42)

    results = {}

    for name, (X, y) in datasets.items():
        print(f"\n--- Dataset: {name} ---")
        print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

        with Timer(f"Linear SVM on {name}"):
            model = train_linear_svm(X, y, C=1.0)

        metrics = evaluate_model(model, X, y)
        results[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_decision_boundary(
            model, X, y,
            title=f"Linear SVM on {name.capitalize()}\n"
                  f"(Accuracy: {metrics['accuracy']:.2%})",
            highlight_sv=True, ax=ax,
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        save_figure(fig, os.path.join(out_dir, f"02_linear_failure_{name}.png"))

    print("\nGenerating combined failure figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap_names = ["Circles", "Moons"]

    for ax, (name, (X, y)), cmap_name in zip(
        axes, datasets.items(), cmap_names
    ):
        model = train_linear_svm(X, y, C=1.0)
        metrics = results[name]
        plot_decision_boundary(
            model, X, y,
            title=f"{cmap_name}\nAccuracy: {metrics['accuracy']:.2%}, "
                  f"F1: {metrics['f1']:.2%}",
            highlight_sv=True, ax=ax,
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    fig.suptitle(
        "Linear SVM Fails on Nonlinear Data",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "02_linear_failure_combined.png"))

    print(f"\nAll figures saved to {out_dir}")
    print("Experiment 2 complete.")


if __name__ == "__main__":
    main()
