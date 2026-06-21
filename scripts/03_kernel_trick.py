"""
Experiment 3: The Kernel Trick.

Trains Polynomial and RBF kernel SVMs on nonlinear datasets
(circles, moons) and compares against Linear SVM.

Demonstrates dramatic improvement from kernelized methods.

Outputs:
  figures/decision_boundaries/03_kernel_trick_circles.png
  figures/decision_boundaries/03_kernel_trick_moons.png
  figures/decision_boundaries/03_kernel_trick_combined.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rkhs_kernel_methods.datasets import make_nonlinear_datasets
from rkhs_kernel_methods.models import train_kernel_svm
from rkhs_kernel_methods.visualization import (
    set_style,
    plot_kernel_comparison,
    save_figure,
)
from rkhs_kernel_methods.evaluation import compare_kernels
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


def main() -> None:
    set_style()
    set_random_seed(42)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE, "figures", "decision_boundaries")
    ensure_dir(out_dir)

    print("=" * 60)
    print("EXPERIMENT 3: The Kernel Trick")
    print("=" * 60)

    datasets = make_nonlinear_datasets(n_samples=300, random_state=42)
    kernels = ["linear", "poly", "rbf"]

    for name, (X, y) in datasets.items():
        print(f"\n{'='*40}")
        print(f"Dataset: {name}")
        print(f"{'='*40}")

        results_df = compare_kernels(X, y, kernels, C=1.0)
        print(results_df.to_string())

        with Timer(f"Training all kernels on {name}"):
            models = {}
            for kernel in kernels:
                models[kernel] = train_kernel_svm(
                    X, y, kernel=kernel, C=1.0,
                )

        titles = [
            f"Linear SVM\nAcc: {results_df.loc[results_df['kernel'] == 'linear', 'accuracy'].values[0]:.2%}",
            f"Polynomial SVM (d=3)\nAcc: {results_df.loc[results_df['kernel'] == 'poly', 'accuracy'].values[0]:.2%}",
            f"RBF (Gaussian) SVM\nAcc: {results_df.loc[results_df['kernel'] == 'rbf', 'accuracy'].values[0]:.2%}",
        ]

        fig = plot_kernel_comparison(models, X, y, titles=titles)
        fig.suptitle(
            f"Kernel Trick on {name.capitalize()}\n"
            "From Linear Failure to Nonlinear Success",
            fontsize=16, fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, os.path.join(out_dir, f"03_kernel_trick_{name}.png"))

    print("\nGenerating combined overview...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row, (name, (X, y)) in enumerate(datasets.items()):
        for col, kernel in enumerate(kernels):
            ax = axes[row, col]
            model = train_kernel_svm(X, y, kernel=kernel, C=1.0)
            from rkhs_kernel_methods.evaluation import evaluate_model

            metrics = evaluate_model(model, X, y)
            from rkhs_kernel_methods.visualization import plot_decision_boundary

            plot_decision_boundary(
                model, X, y,
                title=f"{kernel.upper()} | {name}\nAcc={metrics['accuracy']:.2%}",
                highlight_sv=True, ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")

    fig.suptitle(
        "Kernel Trick: Complete Comparison",
        fontsize=18, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "03_kernel_trick_combined.png"))

    print(f"\nAll figures saved to {out_dir}")
    print("Experiment 3 complete.")


if __name__ == "__main__":
    main()
