# Experiment Report: Kernel Methods and Reproducing Kernel Hilbert Spaces

## Overview

This project provides a self-contained exploration of Reproducing Kernel Hilbert Spaces (RKHS) and their application to Support Vector Machine (SVM) classification. It traces the full intellectual arc from linear maximum-margin classification, through the fundamental limitation of linear methods on nonlinear data, to the kernel trick as a computational breakthrough, and finally to the rigorous mathematical framework of RKHS theory and Mercer's theorem.

All experiments are reproducible with fixed random seeds and produce publication-quality figures. The entire pipeline can be executed with a single command.

---

## Key Concepts Covered

### 1. Linear SVM and Maximum Margin Classification

The Support Vector Machine finds the hyperplane that maximizes the margin — the distance between the decision boundary and the nearest points of each class. Points on the margin boundary are called *support vectors*, and they alone determine the classifier. This sparsity property distinguishes SVMs from probabilistic classifiers like logistic regression, which use all data points equally. The primal optimization problem

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(\langle w, x_i \rangle + b) \geq 1$$

is a convex quadratic program with a unique global solution.

### 2. The Failure of Linear Methods

When data exhibits nonlinear structure — concentric circles, interlocking moons — linear classifiers fail regardless of optimization quality. Experiment 2 demonstrates this: a linear SVM achieves approximately 55% accuracy on the circles dataset (barely above random) and approximately 86% on the moons dataset (substantially suboptimal). This is not a failure of training but a fundamental limitation of the linear hypothesis class.

### 3. The Kernel Trick

The kernel trick replaces inner products $\langle x_i, x_j \rangle$ in the SVM dual formulation with a kernel evaluation $k(x_i, x_j)$ that corresponds to an inner product in a (possibly infinite-dimensional) feature space:

$$k(x, y) = \langle \Phi(x), \Phi(y) \rangle_{\mathcal{H}}$$

The critical insight is that $\Phi(x)$ is never computed explicitly. The polynomial kernel $k(x,y) = (\gamma\langle x,y\rangle + r)^d$ maps to a feature space of combinatorial dimension (exponential in $d$), yet evaluates in $O(p)$ time. The RBF kernel $k(x,y) = \exp(-\gamma\|x-y\|^2)$ maps to an *infinite-dimensional* feature space, yet evaluates in $O(p)$ time as well. This is the central computational insight of kernel methods.

### 4. Reproducing Kernel Hilbert Spaces

An RKHS is a Hilbert space $\mathcal{H}$ of functions where point evaluation is a continuous linear functional. The *reproducing property*

$$f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}}$$

is the defining characteristic: the kernel function $k(\cdot, x)$ acts as the Riesz representer of the evaluation functional at $x$. This connects the computational trick (kernel evaluations) to functional analysis (Hilbert space theory) and guarantees that kernel-based learning algorithms behave well with respect to convergence, regularization, and generalization.

### 5. Mercer's Theorem

Mercer's theorem provides the necessary and sufficient conditions for a kernel function to define a valid RKHS: if $k$ is continuous, symmetric, and its Gram matrix $K_{ij} = k(x_i, x_j)$ is positive semi-definite for any finite set of points, then there exists a feature map $\Phi$ into a Hilbert space such that $k(x,y) = \langle\Phi(x), \Phi(y)\rangle$. The theorem also provides the eigen-decomposition

$$k(x,y) = \sum_{i=1}^{\infty} \lambda_i e_i(x) e_i(y), \quad \lambda_i \geq 0$$

which gives a constructive definition of the feature map. The rapid decay of eigenvalues observed in practice explains why kernel methods generalize well despite operating in high-dimensional spaces.

---

## Experimental Findings

### Experiment 1: Linear SVM vs Logistic Regression
- **Dataset:** Linearly separable 2D blobs (n=300)
- **Finding:** Both classifiers achieve near-perfect accuracy, but SVM produces a maximal-margin boundary with a well-defined geometric interpretation. The support vectors (typically < 20 points) completely determine the boundary.
- **Figures:** `figures/margins/01_logistic_vs_svm.png`, `01_svm_margin.png`, `01_support_vectors.png`

### Experiment 2: Failure of Linear Models
- **Datasets:** Concentric circles, two moons (n=300 each)
- **Finding:** Linear SVM accuracy drops to ~55% on circles and ~86% on moons. The decision boundaries cut straight through nonlinear structures.
- **Figures:** `figures/decision_boundaries/02_linear_failure_circles.png`, `02_linear_failure_moons.png`, `02_linear_failure_combined.png`

### Experiment 3: The Kernel Trick in Action
- **Kernels compared:** Linear, polynomial (d=3), RBF
- **Finding:** Polynomial kernel lifts accuracy to ~98%+ on both nonlinear datasets. RBF kernel reaches ~99%+ on all datasets. The improvement from linear to RBF is the difference between useless and near-perfect.
- **Figures:** `figures/decision_boundaries/03_kernel_trick_circles.png`, `03_kernel_trick_moons.png`, `03_kernel_trick_combined.png`

### Experiment 4: Systematic Kernel Comparison
- **Scope:** 3 datasets × 3 kernels × 4 metrics (accuracy, precision, recall, F1)
- **Key result:** The RBF kernel dominates across all datasets and all metrics. The linear kernel is competitive only on linearly separable data. The polynomial kernel is a strong middle ground.
- **Figures:** `figures/kernels/04_kernel_comparison_metrics.png`, `04_kernel_comparison_heatmap.png`
- **Data:** `results/kernel_comparison.csv`, `results/kernel_comparison.md`

### Experiment 5: RKHS Educational Visualizations
- **Demonstrations:** Polynomial feature map (2D → 3D), implicit embedding via Gram matrix, kernel-inner product equivalence verification, RKHS geometry (kernel functions, spectrum, reproducing property), and a comprehensive three-panel intuition summary.
- **Figures:** `figures/rkhs/05_feature_map.png`, `05_implicit_embedding.png`, `05_kernel_inner_product.png`, `05_rkhs_geometry.png`, `05_rkhs_intuition_demo.png`

### Summary Figures
- **Composite overviews:** Title card, annotated SVM concept, kernel trick comparison grid, RKHS concept map, key findings summary.
- **Figures:** `reports/figures/title_card.png`, `svm_concept.png`, `kernel_trick_flow.png`, `rkhs_summary.png`, `conclusions.png`

### Summary Results Table

| Dataset | Linear SVM | Polynomial SVM (d=3) | RBF SVM |
|---------|-----------|----------------------|---------|
| Linear  | ~1.00     | ~1.00                | ~1.00   |
| Moons   | ~0.86     | ~0.98                | ~0.98   |
| Circles | ~0.55     | ~0.99                | ~0.99   |

---

## Figures Inventory

All figures are generated at 150 DPI with consistent styling (color palette: #4472C4 blue / #ED7D31 orange / #70AD47 green) and publication-quality formatting.

| Directory | Figures | Purpose |
|-----------|---------|---------|
| `figures/margins/` | 3 figures | Logistic regression vs SVM, margin visualization, support vectors |
| `figures/decision_boundaries/` | 6 figures | Linear failure on circles/moons, kernel trick success, combined grids |
| `figures/kernels/` | 2 figures | Grouped bar chart of all metrics, accuracy heatmap |
| `figures/rkhs/` | 5 figures | Feature map, Gram matrix, kernel-inner product, RKHS geometry, intuition |
| `reports/figures/` | 5 figures | Title card, SVM concept, kernel flow, RKHS diagram, findings summary |

---

## How to Use This Repository

### Setup

```bash
pip install -r requirements.txt
```

### Running Experiments

Individual experiments:

```bash
python scripts/01_linear_svm.py
python scripts/02_linear_failure.py
python scripts/03_kernel_trick.py
python scripts/04_kernel_comparison.py
python scripts/05_rkhs_visualizations.py
python scripts/05b_generate_summary_figures.py
```

All experiments at once:

```bash
python scripts/06_generate_all_figures.py
```

### Key Source Modules

| Module | Purpose |
|--------|---------|
| `src/rkhs_kernel_methods/datasets.py` | Dataset generators (linear, circles, moons) |
| `src/rkhs_kernel_methods/kernels.py` | Kernel functions (linear, polynomial, RBF) and Gram matrix |
| `src/rkhs_kernel_methods/models.py` | Training wrappers for SVM and logistic regression |
| `src/rkhs_kernel_methods/evaluation.py` | Metrics, comparison tables, result export |
| `src/rkhs_kernel_methods/theory.py` | Margin width, Mercer condition verification, hinge loss |
| `src/rkhs_kernel_methods/visualization.py` | Plotting utilities, decision boundary rendering |
| `src/rkhs_kernel_methods/rkhs.py` | RKHS demonstrations (feature maps, Gram matrices, reproducing property) |
| `src/rkhs_kernel_methods/utils.py` | Directory creation, timing, random seed |

## Future Extensions

This project provides a foundation that can be extended in several directions:

- **Gaussian Processes:** The connection between kernel methods and GPs — the kernel becomes the covariance function of a prior over functions.
- **Kernel PCA and Spectral Methods:** Extending the kernel trick beyond classification to unsupervised dimensionality reduction and clustering.
- **Multiple Kernel Learning (MKL):** Learning an optimal convex combination of kernels for heterogeneous data sources.
- **Deep Kernel Learning:** Using neural networks to learn the feature map while using kernel methods for the final layer.
- **Neural Tangent Kernel (NTK):** Exploring the connection between infinitely-wide neural networks and kernel methods.
- **Scalability:** Investigating approximation methods (Nyström, random Fourier features) for large-scale datasets.
- **Real-World Benchmarks:** Applying the kernel comparison framework to benchmark datasets (UCI, OpenML).
- **Hyperparameter Sensitivity:** Studying the effect of C, gamma, and degree on decision boundaries and generalization.

---

*Generated by `scripts/06_generate_all_figures.py` — all results are fully reproducible.*
