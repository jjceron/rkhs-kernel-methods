"""
Dataset generation utilities for SVM and kernel method experiments.

Provides wrappers around scikit-learn dataset generators with
consistent interfaces for reproducible experiments.
"""

import numpy as np
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    make_classification,
)
from typing import Dict, Tuple


def make_linearly_separable(
    n_samples: int = 300,
    n_features: int = 2,
    cluster_std: float = 0.60,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a linearly separable 2D dataset using Gaussian blobs.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    n_features : int, default=2
        Number of features (must be 2 for visualization).
    cluster_std : float, default=0.60
        Standard deviation of the clusters.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary labels (-1, +1).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def make_moons_dataset(
    n_samples: int = 300,
    noise: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a two-moons dataset (nonlinear).

    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    noise : float, default=0.15
        Standard deviation of Gaussian noise added to data.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    y = np.where(y == 0, -1, 1)
    return X, y


def make_circles_dataset(
    n_samples: int = 300,
    noise: float = 0.10,
    factor: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a concentric circles dataset (nonlinear).

    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    noise : float, default=0.10
        Gaussian noise standard deviation.
    factor : float, default=0.5
        Scale factor between inner and outer circle.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    """
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def make_nonlinear_datasets(
    n_samples: int = 300,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a collection of nonlinear benchmark datasets.

    Parameters
    ----------
    n_samples : int, default=300
        Number of samples per dataset.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    datasets : dict
        Dictionary mapping dataset names to (X, y) tuples.
        Keys: 'moons', 'circles'.
    """
    return {
        "moons": make_moons_dataset(
            n_samples=n_samples, random_state=random_state
        ),
        "circles": make_circles_dataset(
            n_samples=n_samples, random_state=random_state
        ),
    }


def load_all_datasets(
    n_samples: int = 300, random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all datasets used in the project.

    Parameters
    ----------
    n_samples : int, default=300
        Number of samples for each dataset.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    datasets : dict
        Dictionary with keys 'linear', 'moons', 'circles' mapping to (X, y).
    """
    X_lin, y_lin = make_linearly_separable(
        n_samples=n_samples, random_state=random_state
    )
    X_moons, y_moons = make_moons_dataset(
        n_samples=n_samples, random_state=random_state
    )
    X_circles, y_circles = make_circles_dataset(
        n_samples=n_samples, random_state=random_state
    )
    return {
        "linear": (X_lin, y_lin),
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
    }
