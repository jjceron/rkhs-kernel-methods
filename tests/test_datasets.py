"""Tests for dataset generation utilities."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from rkhs_kernel_methods.datasets import (
    make_linearly_separable,
    make_moons_dataset,
    make_circles_dataset,
    make_nonlinear_datasets,
    load_all_datasets,
)


class TestDatasetGeneration:
    def test_linearly_separable_shape(self):
        X, y = make_linearly_separable(n_samples=300)
        assert X.shape == (300, 2)
        assert y.shape == (300,)

    def test_linearly_separable_labels(self):
        _, y = make_linearly_separable(n_samples=200)
        assert set(np.unique(y)) <= {-1, 1}

    def test_moons_shape(self):
        X, y = make_moons_dataset(n_samples=200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_circles_shape(self):
        X, y = make_circles_dataset(n_samples=200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_nonlinear_datasets(self):
        datasets = make_nonlinear_datasets(n_samples=150)
        assert "moons" in datasets
        assert "circles" in datasets
        for name, (X, y) in datasets.items():
            assert X.shape == (150, 2)
            assert y.shape == (150,)

    def test_load_all_datasets(self):
        datasets = load_all_datasets(n_samples=100)
        assert "linear" in datasets
        assert "moons" in datasets
        assert "circles" in datasets
        for name, (X, y) in datasets.items():
            assert X.shape == (100, 2)
            assert y.shape == (100,)

    def test_reproducibility(self):
        X1, y1 = make_linearly_separable(random_state=42)
        X2, y2 = make_linearly_separable(random_state=42)
        assert np.allclose(X1, X2)
        assert np.array_equal(y1, y2)
