"""
Experiment 1: Linear Classification with SVM and Logistic Regression.

Generates linearly separable data, trains both a Logistic Regression
classifier and a Linear SVM, and visualizes:
  - Decision boundaries
  - Maximum margin (SVM)
  - Support vectors

Outputs:
  figures/margins/01_logistic_vs_svm.png
  figures/margins/01_svm_margin.png
  figures/margins/01_support_vectors.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rkhs_kernel_methods.datasets import make_linearly_separable
from rkhs_kernel_methods.models import (
    train_linear_svm,
    train_logistic_regression,
)
from rkhs_kernel_methods.visualization import (
    set_style,
    plot_decision_boundary,
    plot_margin,
    plot_support_vectors,
    save_figure,
)
from rkhs_kernel_methods.theory import compute_margin_width
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


def main() -> None:
    set_style()
    set_random_seed(42)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE, "figures", "margins")
    ensure_dir(out_dir)

    print("=" * 60)
    print("EXPERIMENT 1: Linear SVM vs Logistic Regression")
    print("=" * 60)

    X, y = make_linearly_separable(n_samples=300, cluster_std=0.60, random_state=42)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y == 1)}")

    with Timer("Logistic Regression"):
        lr_model = train_logistic_regression(X, y, C=1.0)

    with Timer("Linear SVM"):
        svm_model = train_linear_svm(X, y, C=1.0)

    margin = compute_margin_width(svm_model)
    n_sv = len(svm_model.support_vectors_)
    print(f"\nSVM margin width: {margin:.4f}")
    print(f"Number of support vectors: {n_sv}")

    print("\nGenerating comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_decision_boundary(
        lr_model, X, y,
        title="Logistic Regression\n(Probabilistic, no margin concept)",
        highlight_sv=False, ax=ax1,
    )
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    plot_decision_boundary(
        svm_model, X, y,
        title=f"Linear SVM (Margin = {margin:.3f})\n(Maximum margin, support vectors)",
        highlight_sv=True, ax=ax2,
    )
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    fig.suptitle(
        "Logistic Regression vs Linear SVM on Linearly Separable Data",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "01_logistic_vs_svm.png"))

    print("\nGenerating SVM margin visualization...")
    fig2, ax = plt.subplots(figsize=(8, 6))
    plot_margin(
        svm_model, X, y,
        title=f"SVM Maximum Margin Classifier\nMargin width = {margin:.3f}",
        ax=ax,
    )
    save_figure(fig2, os.path.join(out_dir, "01_svm_margin.png"))

    print("\nGenerating support vector highlight...")
    fig3, ax = plt.subplots(figsize=(8, 6))
    plot_support_vectors(
        svm_model, X, y,
        title=f"Support Vectors (n={n_sv})\nOnly support vectors define the boundary",
        ax=ax,
    )
    save_figure(fig3, os.path.join(out_dir, "01_support_vectors.png"))

    print(f"\nAll figures saved to {out_dir}")
    print("Experiment 1 complete.")


if __name__ == "__main__":
    main()
