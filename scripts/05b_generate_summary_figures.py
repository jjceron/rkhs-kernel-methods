"""
Summary figure generation for experiment reports and documentation.

Creates high-level composite figures suitable for project documentation,
benchmark summaries, and technical reports. These figures synthesize results
from experiments 1-5 into digestible overview graphics.

Outputs:
  reports/figures/title_card.png
  reports/figures/svm_concept.png
  reports/figures/kernel_trick_flow.png
  reports/figures/rkhs_summary.png
  reports/figures/conclusions.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rkhs_kernel_methods.datasets import (
    make_linearly_separable,
    make_nonlinear_datasets,
)
from rkhs_kernel_methods.models import (
    train_linear_svm,
    train_kernel_svm,
)
from rkhs_kernel_methods.visualization import (
    set_style,
    plot_decision_boundary,
    plot_margin,
    save_figure as viz_save_figure,
)
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


def create_title_card(out_path: str) -> None:
    """Generate a title graphic for the project."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8, 6,
        "Kernel Methods and\nReproducing Kernel Hilbert Spaces",
        transform=ax.transData,
        fontsize=32, fontweight="bold",
        horizontalalignment="center", verticalalignment="center",
    )

    ax.text(
        8, 3.5,
        "From Linear Support Vector Machines to the Kernel Trick",
        transform=ax.transData,
        fontsize=20,
        horizontalalignment="center", verticalalignment="center",
        style="italic",
    )

    ax.text(
        8, 1.5,
        "Theory, Implementation, and Reproducible Experiments",
        fontsize=16, color="gray",
        horizontalalignment="center", verticalalignment="center",
    )

    rect = plt.Rectangle(
        (2, 0.5), 12, 0.08,
        facecolor="#4472C4", edgecolor="none",
        transform=ax.transData,
    )
    ax.add_patch(rect)

    viz_save_figure(fig, out_path)


def create_svm_concept(out_path: str) -> None:
    """Generate an annotated SVM maximum-margin diagram."""
    set_style()
    set_random_seed(42)
    X, y = make_linearly_separable(
        n_samples=200, cluster_std=0.65, random_state=42
    )
    model = train_linear_svm(X, y, C=1.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_margin(model, X, y, title="", ax=ax)

    margin = 2.0 / np.linalg.norm(model.coef_.ravel())
    ax.set_title(
        f"Maximum Margin Classification (Margin = {margin:.3f})",
        fontsize=18, fontweight="bold",
    )

    ax.annotate(
        "Support Vectors\n(define the margin)",
        xy=(model.support_vectors_[0, 0], model.support_vectors_[0, 1]),
        xytext=(
            model.support_vectors_[0, 0] + 1.5,
            model.support_vectors_[0, 1] + 1.0,
        ),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=13, color="red", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.annotate(
        "Margin Boundary\n(Maximized gap)",
        xy=(0.5, 1.8),
        xytext=(-2, 3.5),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        fontsize=13, color="green", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    viz_save_figure(fig, out_path)


def create_kernel_trick_flow(out_path: str) -> None:
    """Generate a side-by-side comparison of kernel methods on nonlinear data."""
    set_style()
    set_random_seed(42)

    datasets = make_nonlinear_datasets(n_samples=200, random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    kernels = ["linear", "poly", "rbf"]
    col_labels = [
        "Linear SVM\n(Fails on nonlinear data)",
        "Polynomial Kernel (d=3)\n(Handles moderate nonlinearity)",
        "RBF (Gaussian) Kernel\n(Near-perfect separation)",
    ]

    for row, (name, (X, y)) in enumerate(datasets.items()):
        for col, kernel in enumerate(kernels):
            ax = axes[row, col]
            model = train_kernel_svm(X, y, kernel=kernel, C=1.0)
            from rkhs_kernel_methods.evaluation import evaluate_model

            metrics = evaluate_model(model, X, y)
            plot_decision_boundary(
                model, X, y,
                title=f"{col_labels[col]}\nAcc: {metrics['accuracy']:.2%}",
                highlight_sv=True, ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")

    fig.suptitle(
        "The Kernel Trick: Transforming Nonlinear Problems\n"
        "into Linearly Separable Ones",
        fontsize=18, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    viz_save_figure(fig, out_path)


def create_rkhs_summary(out_path: str) -> None:
    """Generate an RKHS concept map showing the key ideas and connections."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        7, 7.55,
        "Reproducing Kernel Hilbert Space (RKHS)",
        fontsize=22, fontweight="bold",
        horizontalalignment="center",
    )

    boxes = [
        (1.5, 4.5, 4.5, 2.3,
         "Input Space $\\mathcal{X}$\n\nNonlinear data\n$x \\in \\mathbb{R}^d$",
         "#B4C7E7"),
        (8.0, 4.5, 4.5, 2.3,
         "Feature Space $\\mathcal{H}$\n\n$\\Phi: \\mathcal{X} \\to \\mathcal{H}$\n"
         "Hilbert space of functions",
         "#F4B183"),
        (1.5, 1.2, 4.5, 2.5,
         "Kernel Trick\n\n$k(x, y) = \\langle \\Phi(x), \\Phi(y) \\rangle_{\\mathcal{H}}$\n"
         "Never compute $\\Phi$ explicitly",
         "#C5E0B4"),
        (8.0, 1.2, 4.5, 2.5,
         "Reproducing Property\n\n$f(x) = \\langle f, k(\\cdot, x) \\rangle_{\\mathcal{H}}$\n"
         "Kernel evaluates functions",
         "#D9C2EC"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle(
            (x, y), w, h,
            facecolor=color, edgecolor="gray", linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            fontsize=10.5,
            horizontalalignment="center", verticalalignment="center",
        )

    # Arrow 1: Input Space -> Feature Space (horizontal, top row gap)
    arrow_left = 6.0   # right edge of top-left box
    arrow_right = 8.0  # left edge of top-right box
    arrow_y_top = 5.65  # mid-height of top-row boxes
    ax.annotate(
        "", xy=(arrow_right, arrow_y_top),
        xytext=(arrow_left, arrow_y_top),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.text(7.0, arrow_y_top + 0.28,
            r"$\Phi: \mathcal{X} \to \mathcal{H}$",
            fontsize=11, ha="center", fontweight="bold", color="#333333")

    # Arrow 2: Input Space -> Kernel Trick (vertical, left column)
    arrow_x_left = 3.75
    ax.annotate(
        "", xy=(arrow_x_left, 4.5),
        xytext=(arrow_x_left, 3.7),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.text(arrow_x_left + 0.3, 4.08,
            r"$k(x,y) = \langle\Phi(x),\Phi(y)\rangle$",
            fontsize=9.5, ha="left", va="center", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    # Arrow 3: Kernel Trick -> Reproducing Property (horizontal, bottom row gap)
    arrow_y_bot = 2.45
    ax.annotate(
        "", xy=(arrow_right, arrow_y_bot),
        xytext=(arrow_left, arrow_y_bot),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=2.5,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.text(7.0, arrow_y_bot + 0.30,
            "Mercer's Theorem",
            fontsize=11, ha="center", fontweight="bold", color="#333333")

    # Arrow 4: Feature Space -> Reproducing Property (vertical, right column)
    arrow_x_right = 10.25
    ax.annotate(
        "", xy=(arrow_x_right, 3.7),
        xytext=(arrow_x_right, 4.5),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.text(arrow_x_right + 0.3, 4.08,
            "Reproducing\nProperty",
            fontsize=9.5, ha="left", va="center", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.9))

    viz_save_figure(fig, out_path)


def create_conclusions(out_path: str) -> None:
    """Generate a conclusions summary figure with key findings."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.2, "Key Findings", fontsize=28, fontweight="bold",
            horizontalalignment="center")

    items = [
        "1. Linear SVMs find the maximum-margin separating hyperplane",
        "2. Linear models fail on nonlinear data (circles, moons) — a structural limitation",
        "3. The Kernel Trick implicitly maps data to high-dimensional feature spaces",
        "4. k(x, y) computes inner products in feature space without any explicit map",
        "5. Polynomial kernels capture polynomial decision boundaries efficiently",
        "6. RBF kernels map data into an infinite-dimensional RKHS",
        "7. RKHS provides the rigorous theoretical foundation for all kernel methods",
        "8. Kernel methods remain practical, powerful, and theoretically well-understood",
    ]

    for i, item in enumerate(items):
        ax.text(
            1.5, 6.0 - i * 0.65, item,
            fontsize=15, fontfamily="serif",
            verticalalignment="center",
        )

    ax.text(
        7, 0.8,
        "Kernel methods bridge linear algorithms and nonlinear problems\n"
        "through the mathematics of Reproducing Kernel Hilbert Spaces.",
        fontsize=14, color="gray", style="italic",
        horizontalalignment="center",
    )

    viz_save_figure(fig, out_path)


def main() -> None:
    set_random_seed(42)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE, "reports", "figures")
    ensure_dir(out_dir)

    print("=" * 60)
    print("GENERATING SUMMARY FIGURES")
    print("=" * 60)

    with Timer("Title card"):
        create_title_card(os.path.join(out_dir, "title_card.png"))

    with Timer("SVM concept"):
        create_svm_concept(os.path.join(out_dir, "svm_concept.png"))

    with Timer("Kernel trick flow"):
        create_kernel_trick_flow(
            os.path.join(out_dir, "kernel_trick_flow.png")
        )

    with Timer("RKHS summary"):
        create_rkhs_summary(os.path.join(out_dir, "rkhs_summary.png"))

    with Timer("Conclusions"):
        create_conclusions(os.path.join(out_dir, "conclusions.png"))

    print(f"\nAll summary figures saved to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
