"""
Visualization utilities for SVM decision boundaries, margins,
support vectors, kernel comparisons, and RKHS demonstrations.

All plots use a consistent, publication-quality style via set_style().
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from typing import Optional, Tuple, Any
from sklearn.svm import SVC
import os


def set_style() -> None:
    """Apply a consistent, publication-quality style to all plots."""
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
        }
    )


def _make_meshgrid(
    X: np.ndarray, h: float = 0.02, margin: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2D mesh grid covering the data range with a margin."""
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a 2D dataset with class colors.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Data points.
    y : np.ndarray of shape (n_samples,)
        Labels.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on. Created if None.

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        X[y == -1, 0], X[y == -1, 1], c="#4472C4", marker="o",
        edgecolors="k", s=50, alpha=0.8, label="Class -1",
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1], c="#ED7D31", marker="s",
        edgecolors="k", s=50, alpha=0.8, label="Class +1",
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="best")
    return ax


def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    highlight_sv: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the decision boundary and decision regions of a classifier.

    Parameters
    ----------
    model : estimator
        Trained classifier with predict method.
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    title : str
    highlight_sv : bool, default=True
        If True and model is SVC, highlight support vectors.
    ax : plt.Axes, optional

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    xx, yy = _make_meshgrid(X)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#B4C7E7", "#F4B183"])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light, levels=[-0.5, 0.5, 1.5])

    ax = plot_dataset(X, y, title=title, ax=ax)

    if highlight_sv and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(
            sv[:, 0], sv[:, 1], s=200, linewidths=1.5,
            facecolors="none", edgecolors="k",
            label="Support Vectors",
        )

    ax.legend(loc="best")
    return ax


def plot_margin(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "SVM Maximum Margin",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the decision boundary, margins, and support vectors of a linear SVM.

    Parameters
    ----------
    model : SVC
        Trained linear SVM.
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    title : str
    ax : plt.Axes, optional

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    xx, yy = _make_meshgrid(X)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = plot_dataset(X, y, title=title, ax=ax)

    ax.contour(
        xx, yy, Z, colors="k", levels=[-1, 0, 1],
        alpha=0.7, linestyles=["--", "-", "--"], linewidths=2,
    )

    sv = model.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1], s=220, linewidths=2.0,
        facecolors="none", edgecolors="#228B22",
        label=f"Support Vectors (n={len(sv)})",
    )

    ax.legend(loc="best")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return ax


def plot_support_vectors(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Support Vectors",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Highlight and label support vectors extracted by the SVM.

    Parameters
    ----------
    model : SVC
        Trained SVM.
    X : np.ndarray
    y : np.ndarray
    title : str
    ax : plt.Axes, optional

    Returns
    -------
    ax : plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    sv = model.support_vectors_
    non_sv_mask = np.ones(len(X), dtype=bool)
    if hasattr(model, "support_"):
        non_sv_mask[model.support_] = False

    ax.scatter(
        X[non_sv_mask, 0], X[non_sv_mask, 1],
        c=["#4472C4" if label == -1 else "#ED7D31" for label in y[non_sv_mask]],
        edgecolors="gray", s=40, alpha=0.5, marker="o",
    )
    ax.scatter(
        sv[:, 0], sv[:, 1], s=200, linewidths=2.0,
        facecolors="none", edgecolors="#228B22",
        label=f"Support Vectors (n={len(sv)})",
    )
    ax.scatter(
        X[~non_sv_mask, 0], X[~non_sv_mask, 1],
        c=["#4472C4" if label == -1 else "#ED7D31" for label in y[~non_sv_mask]],
        edgecolors="#228B22", s=80, linewidths=2.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="best")
    return ax


def plot_kernel_comparison(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    titles: Optional[list] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Plot a side-by-side comparison of kernel decision boundaries.

    Parameters
    ----------
    models : dict
        Dictionary mapping kernel name to trained SVC model.
    X : np.ndarray
    y : np.ndarray
    titles : list of str, optional
        Subplot titles.
    figsize : tuple, default=(18, 5)

    Returns
    -------
    fig : plt.Figure
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    keys = list(models.keys())
    if titles is None:
        titles = keys

    for ax, key, title in zip(axes, keys, titles):
        plot_decision_boundary(models[key], X, y, title=title, ax=ax)

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure, filepath: str, close: bool = True
) -> None:
    """
    Save a matplotlib figure to disk, creating directories as needed.

    Parameters
    ----------
    fig : plt.Figure
    filepath : str
    close : bool, default=True
        Close the figure after saving to free memory.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
