"""
Theoretical computations and utility functions for SVM and kernel methods.

Includes margin width computation, Mercer condition verification
via eigenvalue decomposition, and hinge loss computation.
"""

import numpy as np
from numpy.linalg import eigvalsh
from typing import Optional
from rkhs_kernel_methods.kernels import compute_kernel_gram


def compute_margin_width(model) -> float:
    """
    Compute the margin width of a trained linear SVM.

    Margin width = 2 / ||w||, where w is the weight vector.

    Parameters
    ----------
    model : trained linear SVC
        Must have coef_ attribute (linear kernel).

    Returns
    -------
    margin : float
        The width of the margin.
    """
    w = model.coef_.ravel()
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return 0.0
    return 2.0 / w_norm


def verify_mercer_condition(
    X: np.ndarray,
    kernel: str = "rbf",
    tol: float = 1e-10,
    **kwargs,
) -> dict:
    """
    Verify that a kernel matrix satisfies Mercer's condition.

    Mercer's theorem requires the kernel Gram matrix to be positive
    semidefinite (all eigenvalues >= -tol). Also checks symmetry.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    kernel : str, default='rbf'
        Kernel type: 'linear', 'polynomial', 'rbf'.
    tol : float, default=1e-10
        Tolerance for negativity of eigenvalues.
    **kwargs
        Kernel parameters.

    Returns
    -------
    result : dict
        Keys: 'is_symmetric', 'min_eigenvalue', 'is_psd',
              'n_negative_eigenvalues', 'condition_number'.
    """
    K = compute_kernel_gram(X, kernel=kernel, **kwargs)

    is_symmetric = np.allclose(K, K.T, atol=tol)
    eigenvalues = eigvalsh(K)
    min_eig = eigenvalues.min()
    n_neg = int(np.sum(eigenvalues < -tol))
    if min_eig > 0:
        cond_num = float(eigenvalues.max() / min_eig)
    else:
        cond_num = float("inf")

    return {
        "is_symmetric": is_symmetric,
        "min_eigenvalue": float(min_eig),
        "is_psd": min_eig >= -tol,
        "n_negative_eigenvalues": n_neg,
        "condition_number": cond_num,
    }


def compute_gram_matrix_eigenvalues(
    X: np.ndarray,
    kernel: str = "rbf",
    **kwargs,
) -> np.ndarray:
    """
    Compute sorted eigenvalues of a kernel Gram matrix.

    Parameters
    ----------
    X : np.ndarray
    kernel : str
    **kwargs
        Kernel parameters.

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted in descending order.
    """
    K = compute_kernel_gram(X, kernel=kernel, **kwargs)
    eigenvalues = eigvalsh(K)
    return eigenvalues[::-1]


def hinge_loss(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute the hinge loss: L = max(0, 1 - y_true * y_score).

    Parameters
    ----------
    y_true : np.ndarray
        True labels in {-1, +1}.
    y_score : np.ndarray
        Decision function values (signed distance to boundary).

    Returns
    -------
    loss : float
        Mean hinge loss.
    """
    losses = np.maximum(0, 1 - y_true * y_score)
    return float(np.mean(losses))
