"""Entry point for the RKHS Kernel Methods project.

Usage:
    python src/main.py

This runs a quick demo showing the core concepts.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rkhs_kernel_methods.datasets import load_all_datasets
from rkhs_kernel_methods.models import train_linear_svm, train_kernel_svm
from rkhs_kernel_methods.evaluation import evaluate_model, generate_comparison_table
from rkhs_kernel_methods.theory import compute_margin_width, verify_mercer_condition
from rkhs_kernel_methods.utils import set_random_seed, Timer


def main() -> None:
    """Run a quick demonstration of the project's core concepts."""
    set_random_seed(42)

    print("=" * 60)
    print("RKHS Kernel Methods — Quick Demo")
    print("=" * 60)

    datasets = load_all_datasets(n_samples=200, random_state=42)

    print("\n1. Linear SVM on separable data")
    with Timer("Linear SVM"):
        X, y = datasets["linear"]
        svm = train_linear_svm(X, y)
        margin = compute_margin_width(svm)
        metrics = evaluate_model(svm, X, y)
        print(f"   Margin width: {margin:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Support vectors: {len(svm.support_vectors_)}")

    print("\n2. Linear SVM fails on circles")
    with Timer("Linear SVM on circles"):
        Xc, yc = datasets["circles"]
        svm_lin = train_linear_svm(Xc, yc)
        metrics = evaluate_model(svm_lin, Xc, yc)
        print(f"   Accuracy: {metrics['accuracy']:.2%}  (poor!)")

    print("\n3. RBF SVM solves circles perfectly")
    with Timer("RBF SVM on circles"):
        svm_rbf = train_kernel_svm(Xc, yc, kernel="rbf")
        metrics = evaluate_model(svm_rbf, Xc, yc)
        print(f"   Accuracy: {metrics['accuracy']:.2%}  (excellent!)")

    print("\n4. Kernel comparison across all datasets")
    kernels = ["linear", "poly", "rbf"]
    df = generate_comparison_table(datasets, kernels, C=1.0)
    print("\n" + df.to_string())

    print("\n5. Mercer condition verification")
    mercer_kwargs = {
        "linear": {},
        "poly": {"degree": 3, "gamma": 1.0, "coef0": 1.0},
        "polynomial": {"degree": 3, "gamma": 1.0, "coef0": 1.0},
        "rbf": {"gamma": 1.0},
    }
    for kernel in ["linear", "poly", "rbf"]:
        kwargs = mercer_kwargs.get(kernel, {})
        result = verify_mercer_condition(X, kernel=kernel, **kwargs)
        pd_label = "Yes" if result["is_psd"] else "No "
        print(f"   {kernel:>6s}: PSD={pd_label}, "
              f"min_eig={result['min_eigenvalue']:.6f}")

    print("\n" + "=" * 60)
    print("Demo complete. Run scripts/*.py for full experiments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
