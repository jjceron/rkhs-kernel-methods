"""
Kernel function implementations for Support Vector Machines.

Implements standard kernel functions and Gram matrix computation
used in kernelized SVMs. Each kernel computes k(x, y) directly
without requiring explicit feature map construction.
"""

import numpy as np
from typing import Optional


def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """
    Linear kernel: k(x, y) = x · y.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.

    Returns
    -------
    float
        Dot product value.
    """
    return float(np.dot(x, y))


def polynomial_kernel(
    x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0
) -> float:
    """
    Polynomial kernel: k(x, y) = (gamma * (x · y) + coef0)^degree.

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.
    degree : int, default=3
        Degree of the polynomial.
    gamma : float, default=1.0
        Scale factor for the dot product.
    coef0 : float, default=1.0
        Independent term.

    Returns
    -------
    float
        Kernel evaluation.
    """
    return float((gamma * np.dot(x, y) + coef0) ** degree)


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Radial Basis Function (Gaussian) kernel: k(x, y) = exp(-gamma * ||x - y||^2).

    Parameters
    ----------
    x : np.ndarray
        First input vector.
    y : np.ndarray
        Second input vector.
    gamma : float, default=1.0
        Kernel width parameter. gamma = 1 / (2 * sigma^2).

    Returns
    -------
    float
        Kernel evaluation.
    """
    diff = x - y
    return float(np.exp(-gamma * np.dot(diff, diff)))


def kernel_matrix(
    X1: np.ndarray,
    X2: Optional[np.ndarray] = None,
    kernel: str = "rbf",
    **kwargs,
) -> np.ndarray:
    """
    Compute the kernel (Gram) matrix between two sets of points.

    K_{ij} = k(X1_i, X2_j)

    Parameters
    ----------
    X1 : np.ndarray of shape (n1, d)
        First set of input vectors.
    X2 : np.ndarray of shape (n2, d), optional
        Second set of input vectors. If None, X2 = X1.
    kernel : str, default='rbf'
        Kernel type: 'linear', 'polynomial', or 'rbf'.
    **kwargs
        Additional arguments passed to the kernel function.

    Returns
    -------
    K : np.ndarray of shape (n1, n2)
        Kernel matrix.
    """
    if X2 is None:
        X2 = X1

    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))

    kernel_map = {
        "linear": linear_kernel,
        "polynomial": lambda a, b: polynomial_kernel(a, b, **kwargs),
        "poly": lambda a, b: polynomial_kernel(a, b, **kwargs),
        "rbf": lambda a, b: rbf_kernel(a, b, **kwargs),
    }
    if kernel not in kernel_map:
        raise ValueError(
            f"Unknown kernel: {kernel!r}. "
            f"Use one of: {list(kernel_map.keys())}"
        )
    kernel_fn = kernel_map[kernel]

    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])

    return K


def compute_kernel_gram(
    X: np.ndarray, kernel: str = "rbf", **kwargs
) -> np.ndarray:
    """
    Compute the Gram matrix K where K_{ij} = k(x_i, x_j).

    Convenience wrapper that computes the kernel matrix of X with itself.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data.
    kernel : str, default='rbf'
        Kernel type.
    **kwargs
        Kernel parameters.

    Returns
    -------
    K : np.ndarray of shape (n, n)
        Gram matrix.
    """
    return kernel_matrix(X, X2=None, kernel=kernel, **kwargs)
