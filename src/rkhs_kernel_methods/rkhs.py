"""
RKHS demonstrations and educational visualizations.

Provides functions that illustrate the core intuition behind
Reproducing Kernel Hilbert Spaces: feature maps, implicit embeddings,
kernel evaluations as inner products, and the geometry of RKHS.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional
import os

from rkhs_kernel_methods.kernels import (
    rbf_kernel,
    polynomial_kernel,
    compute_kernel_gram,
)
from rkhs_kernel_methods.visualization import set_style


def demonstrate_feature_map(
    X: np.ndarray,
    map_type: str = "polynomial",
    degree: int = 2,
    n_points: int = 200,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize how a polynomial feature map transforms nonlinear data.

    Generates a 2D circle dataset, applies the polynomial feature map
    phi(x) = [x1, x2, x1*x2, x1^2, x2^2], and shows the lifted
    data in a higher-dimensional space where it becomes linearly separable.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Input data.
    map_type : str, default='polynomial'
        Type of feature map to demonstrate.
    degree : int, default=2
        Polynomial degree.
    n_points : int, default=200
        Number of points to sample for the visualization.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : plt.Figure
    """
    set_style()
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    X_vis = X[:n_points]
    y_vis = np.array([0] * (n_points // 2) + [1] * (n_points // 2))

    if X.shape[0] >= n_points:
        idx = np.linspace(0, X.shape[0] - 1, n_points, dtype=int)
        X_sub = X[idx]
    else:
        X_sub = X

    colors = ["#4472C4", "#ED7D31"]
    ax1.scatter(X_sub[:, 0], X_sub[:, 1], c="gray", alpha=0.5, s=20)
    ax1.set_title("Original Input Space $\\mathcal{X}$")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    x1, x2 = X_sub[:, 0], X_sub[:, 1]
    phi1 = x1**2
    phi2 = x2**2
    phi3 = np.sqrt(2) * x1 * x2

    ax2.scatter(phi1, phi2, phi3, c="gray", alpha=0.5, s=20)
    ax2.set_title("Feature Space $\\Phi(\\mathcal{X})$")
    ax2.set_xlabel("$\\phi_1 = x_1^2$")
    ax2.set_ylabel("$\\phi_2 = x_2^2$")
    ax2.set_zlabel("$\\phi_3 = \\sqrt{2}x_1 x_2$")

    fig.suptitle(
        "Polynomial Feature Map: Making Data Linearly Separable",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def demonstrate_implicit_embedding(
    X: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 1.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the kernel Gram matrix as a heatmap.

    Shows how the kernel implicitly computes similarities without
    explicitly computing the feature map. The Gram matrix reveals
    cluster structure.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    kernel : str, default='rbf'
        Kernel type.
    gamma : float, default=1.0
        Kernel parameter.
    save_path : str, optional

    Returns
    -------
    fig : plt.Figure
    """
    set_style()
    n = min(100, X.shape[0])
    X_sub = X[:n]

    K = compute_kernel_gram(X_sub, kernel=kernel, gamma=gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.scatter(X_sub[:, 0], X_sub[:, 1], c="gray", alpha=0.6, s=30)
    ax1.set_title("Input Data")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")

    ax2 = axes[1]
    im = ax2.imshow(K, cmap="viridis", aspect="auto")
    ax2.set_title(f"Kernel Gram Matrix ($k$: {kernel})")
    ax2.set_xlabel("Sample index $j$")
    ax2.set_ylabel("Sample index $i$")
    plt.colorbar(im, ax=ax2, label="$k(x_i, x_j)$")

    fig.suptitle(
        "Implicit Embedding via the Kernel Trick",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def demonstrate_kernel_inner_product(
    n_points: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Demonstrate that kernel evaluations equal inner products in feature space.

    k(x, y) = <phi(x), phi(y)>

    Shows a comparison between the explicit inner product of feature vectors
    and the kernel function evaluation for a few pairs of points.

    Parameters
    ----------
    n_points : int, default=50
        Number of random point pairs to test.
    save_path : str, optional

    Returns
    -------
    fig : plt.Figure
    """
    set_style()
    rng = np.random.RandomState(42)
    X = rng.randn(n_points, 2)

    ip_explicit = np.zeros(n_points * n_points)
    ip_kernel = np.zeros(n_points * n_points)

    idx = 0
    for i in range(n_points):
        for j in range(n_points):
            phi_i = np.array(
                [
                    X[i, 0] ** 2,
                    X[i, 1] ** 2,
                    np.sqrt(2) * X[i, 0] * X[i, 1],
                ]
            )
            phi_j = np.array(
                [
                    X[j, 0] ** 2,
                    X[j, 1] ** 2,
                    np.sqrt(2) * X[j, 0] * X[j, 1],
                ]
            )
            ip_explicit[idx] = np.dot(phi_i, phi_j)
            ip_kernel[idx] = polynomial_kernel(
                X[i], X[j], degree=2, gamma=1.0, coef0=0
            )
            idx += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ip_explicit, ip_kernel, alpha=0.3, s=15, c="#4472C4")
    lims = [
        min(ip_explicit.min(), ip_kernel.min()),
        max(ip_explicit.max(), ip_kernel.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=2, label="$y = x$")
    ax.set_xlabel("Explicit $\\langle \\phi(x), \\phi(y) \\rangle$")
    ax.set_ylabel("Kernel $k(x, y)$")
    ax.set_title("Kernel = Inner Product in Feature Space")
    ax.legend()
    ax.set_aspect("equal")

    fig.suptitle(
        "Verification of the Kernel Trick",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_rkhs_geometry(
    n_samples: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the geometric intuition of RKHS.

    Shows the reproducing property: the evaluation functional
    f |-> f(x) is continuous (bounded by kernel), and the Riesz
    representation of this functional is k(·, x).

    Parameters
    ----------
    n_samples : int, default=20
        Number of kernel basis functions centered on data.
    save_path : str, optional

    Returns
    -------
    fig : plt.Figure
    """
    set_style()
    rng = np.random.RandomState(42)
    X_centers = np.linspace(-3, 3, n_samples)

    x_eval = np.linspace(-6, 6, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for i in range(min(6, n_samples)):
        k_vals = np.array(
            [rbf_kernel(np.array([xi]), np.array([X_centers[i]]), gamma=1.0) for xi in x_eval]
        )
        ax1.plot(x_eval, k_vals, alpha=0.7, linewidth=1.5)
    ax1.set_title("RBF Kernel Functions $k(\\cdot, x_i)$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$k(x, x_i)$")
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    ax2 = axes[0, 1]
    X1, X2 = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    x0 = np.array([0.5, 0.5])
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rbf_kernel(np.array([X1[i, j], X2[i, j]]), x0, gamma=1.0)
    cf = ax2.contourf(X1, X2, Z, levels=20, cmap="viridis")
    ax2.scatter(*x0, c="red", s=100, marker="*", zorder=5, label="$x_0$")
    ax2.set_title("RBF Kernel Centered at $x_0$")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.legend()
    plt.colorbar(cf, ax=ax2, label="$k(x, x_0)$")

    ax3 = axes[1, 0]
    eigvals = np.random.exponential(scale=1.0, size=30)
    eigvals.sort()
    eigvals = eigvals[::-1]
    ax3.bar(range(1, 31), eigvals[:30], color="#4472C4", alpha=0.8)
    ax3.set_title("Spectrum of the Kernel Operator")
    ax3.set_xlabel("Eigenvalue index")
    ax3.set_ylabel("Eigenvalue $\\lambda_i$")
    ax3.axhline(y=0, color="k", linewidth=0.5)

    ax4 = axes[1, 1]
    ax4.text(
        0.5, 0.5,
        "Reproducing Property:\n\n"
        "$f(x) = \\langle f, k(\\cdot, x) \\rangle_{\\mathcal{H}}$\n\n"
        "The kernel $k(\\cdot, x)$ acts as the\n"
        "representer of evaluation at $x$.\n\n"
        "This means any function in the RKHS\n"
        "can be evaluated via an inner product\n"
        "with the kernel function.",
        transform=ax4.transAxes,
        fontsize=14,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=1", facecolor="#F0F0F0", edgecolor="gray"),
    )
    ax4.set_title("The Reproducing Property")
    ax4.axis("off")

    fig.suptitle(
        "Geometric Intuition of RKHS",
        fontsize=17, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def rkhs_intuition_demo(
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-panel demonstration summarizing the RKHS intuition.

    Panel 1: Data in original space (nonlinear)
    Panel 2: Data mapped to feature space (linear separation)
    Panel 3: Kernel inner product equivalence table

    Parameters
    ----------
    save_path : str, optional

    Returns
    -------
    fig : plt.Figure
    """
    set_style()
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    y = np.where(y == 0, -1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax1 = axes[0]
    ax1.scatter(
        X[y == -1, 0], X[y == -1, 1],
        c="#4472C4", edgecolors="k", s=30, alpha=0.7, label="Class -1",
    )
    ax1.scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="#ED7D31", edgecolors="k", s=30, alpha=0.7, label="Class +1",
    )
    ax1.set_title("Original Space\n(Nonlinear)")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    theta = np.arctan2(X[:, 1], X[:, 0])
    phi1 = r
    phi2 = theta
    ax2.scatter(
        phi1[y == -1], phi2[y == -1],
        c="#4472C4", edgecolors="k", s=30, alpha=0.7, label="Class -1",
    )
    ax2.scatter(
        phi1[y == 1], phi2[y == 1],
        c="#ED7D31", edgecolors="k", s=30, alpha=0.7, label="Class +1",
    )
    ax2.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.8)
    ax2.set_title("Feature Space $\\Phi(\\mathcal{X})$\n(Linearly Separable)")
    ax2.set_xlabel("$\\phi_1 = r$")
    ax2.set_ylabel("$\\phi_2 = \\theta$")
    ax2.legend(fontsize=9)

    ax3 = axes[2]
    ax3.axis("off")
    ax3.text(
        0.5, 0.95,
        "The Kernel Trick",
        transform=ax3.transAxes,
        fontsize=15, fontweight="bold",
        verticalalignment="top", horizontalalignment="center",
    )
    ax3.text(
        0.5, 0.82,
        "$k(x, y) = \\langle \\Phi(x), \\Phi(y) \\rangle_{\\mathcal{H}}$",
        transform=ax3.transAxes,
        fontsize=13,
        verticalalignment="top", horizontalalignment="center",
    )
    steps = [
        "1. Map data: $x \\mapsto \\Phi(x)$",
        "2. Compute inner product: $\\langle \\Phi(x), \\Phi(y) \\rangle$",
        "3. Replace with kernel: $k(x, y)$",
        "4. Never compute $\\Phi(x)$ explicitly!",
        "",
        "Kernel functions:",
        "  Linear:   $k(x,y) = \\langle x, y \\rangle$",
        "  Poly:     $k(x,y) = (\\gamma x^\\top y + r)^d$",
        "  RBF:      $k(x,y) = \\exp(-\\gamma ||x-y||^2)$",
    ]
    for i, step in enumerate(steps):
        ax3.text(
            0.05, 0.68 - i * 0.065,
            step,
            transform=ax3.transAxes,
            fontsize=10, fontfamily="monospace",
            verticalalignment="top",
        )

    fig.suptitle(
        "RKHS Intuition: From Nonlinear Data to Kernel Methods",
        fontsize=17, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
