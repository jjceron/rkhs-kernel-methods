"""
Reproducing Kernel Hilbert Spaces and Kernel Methods.

A complete educational toolkit for understanding Support Vector Machines,
the Kernel Trick, and Reproducing Kernel Hilbert Spaces (RKHS).
"""

__version__ = "1.0.0"

from rkhs_kernel_methods.datasets import (
    make_linearly_separable,
    make_nonlinear_datasets,
    load_all_datasets,
)
from rkhs_kernel_methods.kernels import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    kernel_matrix,
    compute_kernel_gram,
)
from rkhs_kernel_methods.models import (
    train_linear_svm,
    train_kernel_svm,
    train_logistic_regression,
    get_support_vectors,
    decision_function,
)
from rkhs_kernel_methods.evaluation import (
    evaluate_model,
    compare_kernels,
    generate_comparison_table,
    save_results_csv,
)
from rkhs_kernel_methods.visualization import (
    plot_decision_boundary,
    plot_margin,
    plot_support_vectors,
    plot_dataset,
    plot_kernel_comparison,
    set_style,
)
from rkhs_kernel_methods.rkhs import (
    demonstrate_feature_map,
    demonstrate_implicit_embedding,
    demonstrate_kernel_inner_product,
    visualize_rkhs_geometry,
    rkhs_intuition_demo,
)
from rkhs_kernel_methods.theory import (
    compute_margin_width,
    verify_mercer_condition,
    compute_gram_matrix_eigenvalues,
    hinge_loss,
)
from rkhs_kernel_methods.utils import (
    ensure_dir,
    save_figure,
    set_random_seed,
    Timer,
)

__all__ = [
    # datasets
    "make_linearly_separable",
    "make_nonlinear_datasets",
    "load_all_datasets",
    # kernels
    "linear_kernel",
    "polynomial_kernel",
    "rbf_kernel",
    "kernel_matrix",
    "compute_kernel_gram",
    # models
    "train_linear_svm",
    "train_kernel_svm",
    "train_logistic_regression",
    "get_support_vectors",
    "decision_function",
    # evaluation
    "evaluate_model",
    "compare_kernels",
    "generate_comparison_table",
    "save_results_csv",
    # visualization
    "plot_decision_boundary",
    "plot_margin",
    "plot_support_vectors",
    "plot_dataset",
    "plot_kernel_comparison",
    "set_style",
    # rkhs
    "demonstrate_feature_map",
    "demonstrate_implicit_embedding",
    "demonstrate_kernel_inner_product",
    "visualize_rkhs_geometry",
    "rkhs_intuition_demo",
    # theory
    "compute_margin_width",
    "verify_mercer_condition",
    "compute_gram_matrix_eigenvalues",
    "hinge_loss",
    # utils
    "ensure_dir",
    "save_figure",
    "set_random_seed",
    "Timer",
]
