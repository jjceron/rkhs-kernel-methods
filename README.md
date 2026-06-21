# Kernel Methods and Reproducing Kernel Hilbert Spaces

**A practical and theoretical exploration of Support Vector Machines, Kernel Methods, and Reproducing Kernel Hilbert Spaces.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.4.0-orange.svg)](https://scikit-learn.org/)
[![tests](https://img.shields.io/badge/tests-41%2F41%20passing-brightgreen.svg)]()

--- 

## Overview

This project explores the progression from linear classifiers to kernel-based machine learning methods. Starting from maximum-margin Support Vector Machines, it demonstrates why linear models fail on nonlinear data, how the kernel trick overcomes this limitation, and how Reproducing Kernel Hilbert Spaces (RKHS) provide the rigorous mathematical foundation.

The repository combines theory, implementation, and reproducible experiments in a clean, modular Python package — suitable as a reference, a learning resource, or a starting point for kernel methods research.

```
Linear Models → Linear SVM → Nonlinear Problems → Kernel Trick → RKHS → Kernelized SVMs
```

## Motivation

Many real-world classification problems are not linearly separable. Support Vector Machines address this through two key innovations:

1. **Maximum Margin Classification** — finding the hyperplane that maximally separates classes, yielding better generalization
2. **The Kernel Trick** — implicitly mapping data into high-dimensional spaces where linear separation becomes possible, without ever computing the mapping explicitly

Reproducing Kernel Hilbert Spaces provide the theoretical foundation that explains why the kernel trick is correct. Understanding this connection bridges applied machine learning and functional analysis.

## Learning Objectives

By working through this repository, you will:

- Understand maximum margin classification and the role of support vectors
- Recognize when and why linear classifiers fail on real-world data
- Apply the kernel trick with polynomial and RBF (Gaussian) kernels
- Compare kernel performance systematically using standard metrics
- Develop intuition for Reproducing Kernel Hilbert Spaces and the reproducing property
- Verify Mercer's theorem empirically through Gram matrix analysis
- Run reproducible experiments that produce publication-quality figures

## Repository Structure

```
rkhs-kernel-methods/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── main.py                          # Quick demo entry point
│   └── rkhs_kernel_methods/             # Core package
│       ├── __init__.py                  # Package exports
│       ├── datasets.py                  # Data generation utilities
│       ├── kernels.py                   # Kernel function implementations
│       ├── models.py                    # SVM & logistic regression wrappers
│       ├── evaluation.py                # Metrics & comparison tables
│       ├── visualization.py             # Publication-quality plotting
│       ├── rkhs.py                      # RKHS demonstrations
│       ├── theory.py                    # Margin computation, Mercer checks, hinge loss
│       └── utils.py                     # General utilities
│
├── scripts/                             # Runnable experiment pipeline
│   ├── 01_linear_svm.py                 # Linear SVM vs Logistic Regression
│   ├── 02_linear_failure.py             # Linear models fail on nonlinear data
│   ├── 03_kernel_trick.py               # Polynomial & RBF kernel SVMs
│   ├── 04_kernel_comparison.py          # Systematic kernel comparison with metrics
│   ├── 05_rkhs_visualizations.py        # RKHS educational demonstrations
│   ├── 05b_generate_summary_figures.py  # Summary figures for documentation
│   └── 06_generate_all_figures.py       # Master orchestrator
│
├── notebooks/                           # Interactive Jupyter notebooks
│   ├── 01_linear_svm.ipynb
│   ├── 02_kernel_trick.ipynb
│   ├── 03_rkhs_intuition.ipynb
│   └── 04_kernel_comparison.ipynb
│
├── docs/                                # Theory documentation
│   ├── svm_theory.md
│   ├── kernel_trick.md
│   ├── rkhs.md
│   ├── mercer_theorem.md
│   └── references.md
│
├── reports/                             # Experiment reports and summary figures
│   ├── final_summary.md
│   └── figures/                         # Summary figures for documentation
│
├── figures/                             # Generated experiment figures
│   ├── margins/
│   ├── decision_boundaries/
│   ├── kernels/
│   └── rkhs/
│
├── results/                             # Generated results (CSV, markdown)
└── tests/                               # Unit tests
    ├── test_datasets.py
    ├── test_kernels.py
    ├── test_models.py
    └── test_rkhs.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rkhs-kernel-methods.git
cd rkhs-kernel-methods

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the quick demo
python src/main.py

# Generate all figures (runs experiments 1-5 sequentially)
python scripts/06_generate_all_figures.py

# Run the test suite
pytest tests/ -v
```

## Experiments

### Experiment 1 — Linear Classification
Compares Logistic Regression vs Linear SVM on linearly separable data. Visualizes decision boundaries, maximum margin, and support vectors.

```bash
python scripts/01_linear_svm.py
```

**Output:** `figures/margins/01_logistic_vs_svm.png`, `01_svm_margin.png`, `01_support_vectors.png`

### Experiment 2 — Failure of Linear Models
Demonstrates the fundamental limitation of linear separators on concentric circles and two-moons datasets.

```bash
python scripts/02_linear_failure.py
```

**Output:** `figures/decision_boundaries/02_linear_failure_*.png`

### Experiment 3 — The Kernel Trick
Trains Polynomial and RBF kernel SVMs, demonstrating the dramatic improvement from kernelized methods.

```bash
python scripts/03_kernel_trick.py
```

**Output:** `figures/decision_boundaries/03_kernel_trick_*.png`

### Experiment 4 — Kernel Comparison Study
Systematic comparison of Linear, Polynomial, and RBF kernels across all datasets with comprehensive metrics: accuracy, precision, recall, F1 score.

```bash
python scripts/04_kernel_comparison.py
```

**Output:** `figures/kernels/04_*`, `results/kernel_comparison.csv`, `results/kernel_comparison.md`

### Experiment 5 — RKHS Visualizations
Educational demonstrations of feature maps, implicit embeddings, kernel-inner product equivalence, Gram matrix structure, and RKHS geometry.

```bash
python scripts/05_rkhs_visualizations.py
```

**Output:** `figures/rkhs/05_*.png`

### Summary Figures
Generates composite overview figures suitable for documentation and technical reports.

```bash
python scripts/05b_generate_summary_figures.py
```

**Output:** `reports/figures/*.png`

## Key Results

| Dataset | Linear SVM | Polynomial SVM (d=3) | RBF SVM |
|---------|-----------|---------------------|---------|
| Linear  | 1.00      | 1.00                | 1.00    |
| Moons   | 0.86      | 0.98                | 0.98    |
| Circles | 0.55      | 0.99                | 0.99    |

The RBF kernel achieves near-perfect accuracy on all datasets by implicitly mapping data into an infinite-dimensional RKHS where any finite dataset becomes linearly separable — all without ever computing the infinite-dimensional feature map.

## Figures

All figures are generated automatically by the experiment scripts:

| Directory | Contents |
|-----------|----------|
| `figures/margins/` | SVM margin visualizations, support vectors, logistic regression comparison |
| `figures/decision_boundaries/` | Decision boundaries for linear failure and kernel success |
| `figures/kernels/` | Kernel comparison bar charts and accuracy heatmaps |
| `figures/rkhs/` | Feature maps, Gram matrices, RKHS geometry, reproducing property |
| `reports/figures/` | Summary figures: title card, SVM concept, kernel flow, RKHS diagram, findings |

## Documentation

See [`docs/`](docs/) for in-depth theory documents:

- [`svm_theory.md`](docs/svm_theory.md) — SVM primal and dual formulations, margin theory
- [`kernel_trick.md`](docs/kernel_trick.md) — Feature maps, kernel trick, polynomial and RBF kernels
- [`rkhs.md`](docs/rkhs.md) — Hilbert spaces, reproducing property, representer theorem
- [`mercer_theorem.md`](docs/mercer_theorem.md) — PSD kernels and Mercer's theorem
- [`references.md`](docs/references.md) — Bibliography and further reading

## Tests

```bash
pytest tests/ -v
```

41 tests covering datasets, kernels, model training, Mercer condition verification, hinge loss, and utility functions.

## Technology Stack

- **Python 3.11+**
- **NumPy** — numerical computing
- **SciPy** — scientific computing and eigenvalue decomposition
- **Pandas** — data manipulation and results export
- **scikit-learn** — SVM implementation and dataset generators
- **Matplotlib** — visualizations
- **Seaborn** — statistical plotting

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Scholkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273-297.
- Mercer, J. (1909). *Functions of Positive and Negative Type and Their Connection with the Theory of Integral Equations*. Philosophical Transactions of the Royal Society A.

See [`docs/references.md`](docs/references.md) for the complete bibliography.

## License

MIT License. See [LICENSE](LICENSE) for details.
