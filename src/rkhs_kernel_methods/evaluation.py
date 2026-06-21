"""
Model evaluation and comparison utilities.

Provides functions for computing classification metrics, generating
comparison tables across kernels, and exporting results.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import SVC
from typing import Dict, List, Tuple, Any
import os


def evaluate_model(
    model: Any, X: np.ndarray, y_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics for a trained model.

    Parameters
    ----------
    model : estimator
        Trained classifier with a predict method.
    X : np.ndarray of shape (n_samples, n_features)
        Test data.
    y_true : np.ndarray of shape (n_samples,)
        True labels.

    Returns
    -------
    metrics : dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compare_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernels: List[str],
    C: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare multiple kernel SVMs on a single dataset.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data.
    y : np.ndarray of shape (n_samples,)
        Labels.
    kernels : list of str
        Kernel names to compare (e.g., ['linear', 'poly', 'rbf']).
    C : float, default=1.0
        Regularization parameter.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: kernel, accuracy, precision, recall, f1.
    """
    from rkhs_kernel_methods.models import train_kernel_svm

    results = []
    for kernel in kernels:
        model = train_kernel_svm(
            X, y, kernel=kernel, C=C, random_state=random_state
        )
        metrics = evaluate_model(model, X, y)
        metrics["kernel"] = kernel
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df[["kernel", "accuracy", "precision", "recall", "f1"]]
    return df


def generate_comparison_table(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    kernels: List[str],
    C: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a full comparison table across datasets and kernels.

    Parameters
    ----------
    datasets : dict
        Dict mapping dataset name to (X, y) tuple.
    kernels : list of str
        Kernel names.
    C : float, default=1.0
        Regularization.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    results_df : pd.DataFrame
        Multi-index DataFrame with dataset and kernel levels.
    """
    all_results = []
    for name, (X, y) in datasets.items():
        df = compare_kernels(X, y, kernels, C=C, random_state=random_state)
        df["dataset"] = name
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    return combined.set_index(["dataset", "kernel"])


def save_results_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a results DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Results to save.
    filepath : str
        Output file path. Parent directories are created if needed.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=True)
