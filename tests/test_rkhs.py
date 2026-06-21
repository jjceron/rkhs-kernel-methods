"""Tests for RKHS and theory utilities."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from rkhs_kernel_methods.kernels import (
    polynomial_kernel,
    rbf_kernel,
    compute_kernel_gram,
)
from rkhs_kernel_methods.theory import (
    verify_mercer_condition,
    compute_gram_matrix_eigenvalues,
    hinge_loss,
)
from rkhs_kernel_methods.visualization import set_style
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


class TestKernelInnerProduct:
    """Verify k(x,y) = <phi(x), phi(y)> for the quadratic polynomial kernel."""

    def test_poly2_inner_product_equivalence(self):
        d = 2
        rng = np.random.RandomState(42)
        X = rng.randn(10, d)

        def phi(x):
            return np.array(
                [x[0] ** 2, x[1] ** 2, np.sqrt(2) * x[0] * x[1]]
            )

        for i in range(10):
            for j in range(10):
                kernel_val = polynomial_kernel(
                    X[i], X[j], degree=2, gamma=1.0, coef0=0
                )
                inner_prod = np.dot(phi(X[i]), phi(X[j]))
                assert np.isclose(kernel_val, inner_prod, rtol=1e-10)


class TestMercerCondition:
    def test_rbf_is_psd(self):
        X = np.random.randn(30, 3)
        result = verify_mercer_condition(X, kernel="rbf", gamma=0.5)
        assert result["is_symmetric"]
        assert result["is_psd"]
        assert result["n_negative_eigenvalues"] == 0

    def test_linear_is_psd(self):
        X = np.random.randn(30, 3)
        result = verify_mercer_condition(X, kernel="linear")
        assert result["is_symmetric"]
        assert result["is_psd"]

    def test_polynomial_is_psd(self):
        X = np.random.randn(30, 3)
        result = verify_mercer_condition(
            X, kernel="polynomial", degree=2, gamma=1.0, coef0=1.0
        )
        assert result["is_symmetric"]
        assert result["is_psd"]


class TestGramEigenvalues:
    def test_eigenvalues_sorted_descending(self):
        X = np.random.randn(20, 3)
        eig = compute_gram_matrix_eigenvalues(X, kernel="rbf", gamma=0.5)
        assert len(eig) == 20
        assert np.all(np.diff(eig) <= 0)

    def test_eigenvalues_nonnegative(self):
        X = np.random.randn(20, 3)
        eig = compute_gram_matrix_eigenvalues(X, kernel="rbf", gamma=0.5)
        assert np.all(eig >= -1e-10)


class TestHingeLoss:
    def test_perfect_separation(self):
        y = np.array([1, 1, -1, -1])
        scores = np.array([5.0, 3.0, -4.0, -2.0])
        loss = hinge_loss(y, scores)
        assert loss == 0.0

    def test_misclassification(self):
        y = np.array([1, 1, -1, -1])
        scores = np.array([-1.0, 0.5, 2.0, -3.0])
        loss = hinge_loss(y, scores)
        assert loss > 0.0

    def test_margin_violation(self):
        y = np.array([1, -1])
        scores = np.array([0.5, -0.3])
        loss = hinge_loss(y, scores)
        expected = (max(0, 1 - 0.5) + max(0, 1 - 0.3)) / 2
        assert np.isclose(loss, expected)


class TestUtils:
    def test_ensure_dir(self):
        import tempfile
        import shutil
        tmp = os.path.join(tempfile.gettempdir(), "rkhs_test_dir")
        try:
            result = ensure_dir(tmp)
            assert os.path.isdir(tmp)
            assert result == tmp
        finally:
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)

    def test_set_random_seed(self):
        set_random_seed(42)
        a1 = np.random.rand()
        set_random_seed(42)
        a2 = np.random.rand()
        assert a1 == a2

    def test_timer_context(self):
        with Timer("Test"):
            _ = sum(range(1000))
        assert True
