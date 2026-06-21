"""Tests for model training and evaluation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from rkhs_kernel_methods.datasets import (
    make_linearly_separable,
    make_moons_dataset,
)
from rkhs_kernel_methods.models import (
    train_linear_svm,
    train_kernel_svm,
    train_logistic_regression,
    get_support_vectors,
    decision_function,
)
from rkhs_kernel_methods.evaluation import evaluate_model


class TestModelTraining:
    def setup_method(self):
        self.X, self.y = make_linearly_separable(n_samples=100, random_state=42)

    def test_linear_svm_fits(self):
        model = train_linear_svm(self.X, self.y)
        assert hasattr(model, "support_vectors_")
        assert len(model.support_vectors_) > 0

    def test_linear_svm_predicts(self):
        model = train_linear_svm(self.X, self.y)
        y_pred = model.predict(self.X)
        assert y_pred.shape == self.y.shape

    def test_kernel_svm_rbf(self):
        X, y = make_moons_dataset(n_samples=100, random_state=42)
        model = train_kernel_svm(X, y, kernel="rbf")
        y_pred = model.predict(X)
        assert np.mean(y_pred == y) > 0.7

    def test_kernel_svm_poly(self):
        model = train_kernel_svm(self.X, self.y, kernel="poly", degree=3)
        y_pred = model.predict(self.X)
        assert np.mean(y_pred == self.y) > 0.8

    def test_kernel_svm_linear(self):
        model = train_kernel_svm(self.X, self.y, kernel="linear")
        assert hasattr(model, "support_vectors_")

    def test_kernel_svm_invalid_raises(self):
        import pytest
        with pytest.raises(ValueError):
            train_kernel_svm(self.X, self.y, kernel="unknown")

    def test_logistic_regression_fits(self):
        model = train_logistic_regression(self.X, self.y)
        assert hasattr(model, "coef_")

    def test_logistic_regression_accuracy(self):
        model = train_logistic_regression(self.X, self.y)
        y_pred = model.predict(self.X)
        assert np.mean(y_pred == self.y) > 0.8

    def test_get_support_vectors(self):
        model = train_linear_svm(self.X, self.y)
        sv_X, sv_y = get_support_vectors(model)
        assert sv_X.shape[1] == 2
        assert len(sv_y) == len(sv_X)

    def test_decision_function(self):
        model = train_linear_svm(self.X, self.y)
        scores = decision_function(model, self.X)
        assert scores.shape == (100,)

    def test_evaluate_model(self):
        model = train_linear_svm(self.X, self.y)
        metrics = evaluate_model(model, self.X, self.y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
