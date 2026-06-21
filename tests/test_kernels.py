"""Tests for kernel function implementations."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from rkhs_kernel_methods.kernels import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    kernel_matrix,
    compute_kernel_gram,
)


class TestKernelFunctions:
    def test_linear_kernel(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert np.isclose(linear_kernel(x, y), 11.0)

    def test_linear_kernel_identity(self):
        x = np.array([1.0, 1.0])
        assert np.isclose(linear_kernel(x, x), 2.0)

    def test_polynomial_kernel(self):
        x = np.array([1.0, 0.0])
        y = np.array([1.0, 0.0])
        val = polynomial_kernel(x, y, degree=2, gamma=1.0, coef0=1.0)
        assert np.isclose(val, 4.0)

    def test_rbf_kernel_self(self):
        x = np.array([1.0, 2.0])
        assert np.isclose(rbf_kernel(x, x, gamma=1.0), 1.0)

    def test_rbf_kernel_range(self):
        x = np.array([0.0, 0.0])
        y = np.array([10.0, 0.0])
        val = rbf_kernel(x, y, gamma=1.0)
        assert 0.0 <= val <= 1.0

    def test_rbf_kernel_symmetry(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        assert np.isclose(rbf_kernel(x, y), rbf_kernel(y, x))

    def test_kernel_matrix_shape(self):
        X = np.random.randn(20, 5)
        K = kernel_matrix(X, kernel="linear")
        assert K.shape == (20, 20)

    def test_kernel_matrix_symmetry(self):
        X = np.random.randn(15, 3)
        K = kernel_matrix(X, kernel="rbf", gamma=0.5)
        assert np.allclose(K, K.T)

    def test_kernel_matrix_two_sets(self):
        X1 = np.random.randn(10, 3)
        X2 = np.random.randn(15, 3)
        K = kernel_matrix(X1, X2, kernel="linear")
        assert K.shape == (10, 15)

    def test_compute_kernel_gram(self):
        X = np.random.randn(20, 4)
        K = compute_kernel_gram(X, kernel="rbf", gamma=0.5)
        assert K.shape == (20, 20)
        assert np.allclose(K, K.T)

    def test_kernel_matrix_positive_diagonal(self):
        X = np.random.randn(10, 3)
        for kernel in ["linear", "rbf"]:
            K = compute_kernel_gram(X, kernel=kernel, gamma=0.5)
            assert np.all(np.diag(K) >= 0)
