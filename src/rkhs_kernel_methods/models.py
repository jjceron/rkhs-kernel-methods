"""
Model training wrappers for SVM and logistic regression.

Provides thin, reproducible wrappers around scikit-learn estimators
with consistent interfaces and support vector extraction.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Optional, Any


def train_linear_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> SVC:
    """
    Train a linear SVM classifier.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    y : np.ndarray of shape (n_samples,)
        Labels in {-1, +1} or {0, 1}.
    C : float, default=1.0
        Regularization parameter.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    model : SVC
        Trained scikit-learn SVM with linear kernel.
    """
    model = SVC(kernel="linear", C=C, random_state=random_state)
    model.fit(X, y)
    return model


def train_kernel_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    random_state: int = 42,
) -> SVC:
    """
    Train a kernelized SVM classifier.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    y : np.ndarray of shape (n_samples,)
        Labels.
    kernel : str, default='rbf'
        Kernel type: 'linear', 'poly', 'rbf'.
    C : float, default=1.0
        Regularization parameter.
    gamma : float, default=1.0
        Kernel coefficient for 'rbf' and 'poly'.
    degree : int, default=3
        Degree for 'poly' kernel.
    coef0 : float, default=1.0
        Independent term for 'poly' kernel.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    model : SVC
        Trained kernel SVM.
    """
    if kernel == "linear":
        model = SVC(kernel="linear", C=C, random_state=random_state)
    elif kernel == "poly":
        model = SVC(
            kernel="poly",
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            random_state=random_state,
        )
    elif kernel == "rbf":
        model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=random_state)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    model.fit(X, y)
    return model


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression classifier.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    y : np.ndarray of shape (n_samples,)
        Labels.
    C : float, default=1.0
        Inverse regularization strength.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    """
    model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
    model.fit(X, y)
    return model


def get_support_vectors(model: SVC) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract support vectors and their labels from a trained SVM.

    Parameters
    ----------
    model : SVC
        Trained scikit-learn SVM.

    Returns
    -------
    sv_X : np.ndarray of shape (n_sv, n_features)
        Support vector coordinates.
    sv_y : np.ndarray of shape (n_sv,)
        Support vector labels.
    """
    sv_X = model.support_vectors_
    sv_y = np.sign(model.dual_coef_[0])
    return sv_X, sv_y


def decision_function(
    model: SVC, X: np.ndarray
) -> np.ndarray:
    """
    Compute the decision function values for a trained SVM.

    Parameters
    ----------
    model : SVC
        Trained SVM.
    X : np.ndarray of shape (n_samples, n_features)
        Input points.

    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
        Signed distance to the decision boundary.
    """
    return model.decision_function(X)
