# The Kernel Trick

## 1. The Problem with Linearity

Linear classifiers are appealing — they are simple, interpretable, and admit convex objective functions. But they are fundamentally limited: a hyperplane cannot separate data that are not linearly separable. Consider the classic XOR problem in $\mathbb{R}^2$:

```
        +          -
            (0,1)   (1,1)
            ~~~~~~~~~~
            (0,0)   (1,0)
        -          +
```

No line can separate the classes {(0,0), (1,1)} from {(0,1), (1,0)}. Yet these points are trivially separable by a nonlinear decision surface.

**The core idea of kernel methods:** map the data into a higher-dimensional space where linear separability becomes possible, then apply the linear SVM in that space — without ever explicitly constructing the high-dimensional representation.

---

## 2. Feature Maps: $\varphi: \mathcal{X} \to \mathcal{F}$

A **feature map** is a function

$$\varphi: \mathcal{X} \to \mathcal{F}$$

where $\mathcal{X} \subseteq \mathbb{R}^d$ is the **input space** and $\mathcal{F}$ is the **feature space** (an inner product space, often $\mathbb{R}^p$ with $p \gg d$ or even infinite-dimensional).

### Example 1: Polynomial feature map (degree 2, $\mathbb{R}^2$)

$$\varphi(\mathbf{x}) = \varphi(x_1, x_2) = (x_1^2,\; \sqrt{2}x_1 x_2,\; x_2^2) \in \mathbb{R}^3$$

A linear hyperplane in $\mathbb{R}^3$ corresponds to a quadratic decision boundary in the original $\mathbb{R}^2$.

### Example 2: Augmented polynomial map (degree 2, with linear terms)

$$\varphi(\mathbf{x}) = (x_1^2,\; x_2^2,\; \sqrt{2}x_1 x_2,\; \sqrt{2}x_1,\; \sqrt{2}x_2,\; 1) \in \mathbb{R}^6$$

The $\sqrt{2}$ factors ensure that $\langle \varphi(\mathbf{x}), \varphi(\mathbf{z}) \rangle = (\mathbf{x} \cdot \mathbf{z} + 1)^2$.

### Example 3: Infinite-dimensional feature map for the RBF kernel

The Gaussian (RBF) kernel

$$k(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right)$$

can be expressed as an inner product in an infinite-dimensional feature space:

$$\varphi(\mathbf{x}) = e^{-\|\mathbf{x}\|^2/2\sigma^2} \left( 1, \sqrt{\frac{1}{1!\sigma^2}} x_1, \dots, \sqrt{\frac{1}{k!\sigma^{2k}}} x_1^k, \dots \right)$$

Explicit construction of this vector is practically impossible — yet the kernel $k(\mathbf{x}, \mathbf{z})$ can be evaluated in $O(d)$ time. This is the kernel trick.

---

## 3. The Computational Problem

If we explicitly compute $\varphi(\mathbf{x})$ and then take inner products, the cost can be prohibitive:

| Kernel type | $\dim(\mathcal{F})$ | Cost of explicit $\varphi$ |
|-------------|---------------------|----------------------------|
| Polynomial degree $m$ in $\mathbb{R}^d$ | $\binom{d+m}{m} = O(d^m)$ | $O(d^m)$ |
| RBF/Gaussian | $\infty$ | Impossible |
| String kernel (all substrings) | Exponential in string length | Exponential |

Even for moderate $d$ and $m$ (e.g., $d=100$, $m=3$), $\dim(\mathcal{F}) \approx 176,851$ — storing and computing with such vectors is expensive.

Moreover, many learning algorithms depend on the data only through inner products. In the SVM dual, for example, we only need $\varphi(\mathbf{x}_i) \cdot \varphi(\mathbf{x}_j)$ — never the individual vectors $\varphi(\mathbf{x}_i)$ in isolation.

---

## 4. The Kernel Trick

A **kernel** is a function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ that computes

$$k(\mathbf{x}, \mathbf{z}) = \langle \varphi(\mathbf{x}), \varphi(\mathbf{z}) \rangle_{\mathcal{F}}$$

for some feature map $\varphi: \mathcal{X} \to \mathcal{F}$ into an inner product space $\mathcal{F}$.

The **kernel trick** is the observation that we can:

1. Replace every inner product $\mathbf{x} \cdot \mathbf{z}$ in a linear algorithm with $k(\mathbf{x}, \mathbf{z})$.
2. This implicitly computes the linear algorithm in $\mathcal{F}$ without ever explicitly constructing $\varphi(\mathbf{x})$.

Formally, for any algorithm whose training and prediction steps depend on data only through inner products, applying the kernel trick yields a nonlinear generalization.

### SVM with Kernels

The dual SVM with kernels becomes:

$$\max_{\boldsymbol{\alpha}} \; \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \, k(\mathbf{x}_i, \mathbf{x}_j)$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$. The decision function:

$$f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i=1}^n \alpha_i y_i \, k(\mathbf{x}_i, \mathbf{x}) + b \right)$$

The **kernel matrix** (or Gram matrix) $\mathbf{K}$ has entries $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$. The dual objective is a quadratic form in $\mathbf{K}$.

---

## 5. Common Kernel Functions

### 5.1 Linear Kernel

$$k(\mathbf{x}, \mathbf{z}) = \mathbf{x} \cdot \mathbf{z}$$

This is the trivial kernel — it recovers the standard linear SVM. The feature map is the identity $\varphi(\mathbf{x}) = \mathbf{x}$.

### 5.2 Polynomial Kernel (Homogeneous)

$$k(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^{m}, \quad m \in \mathbb{N}$$

The feature space consists of all monomials of degree exactly $m$. For $m=2$, $d=2$:

$$(\mathbf{x} \cdot \mathbf{z})^2 = (x_1 z_1 + x_2 z_2)^2 = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$

which equals $\langle (x_1^2,\; \sqrt{2}x_1 x_2,\; x_2^2),\; (z_1^2,\; \sqrt{2}z_1 z_2,\; z_2^2) \rangle$.

### 5.3 Polynomial Kernel (Inhomogeneous)

$$k(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z} + c)^m, \quad c \geq 0$$

The constant $c$ trades off the influence of higher-order versus lower-order terms. When $c > 0$, the feature space includes all monomials up to degree $m$.

**Feature space dimension:** For inputs in $\mathbb{R}^d$ and degree $m$, the feature space has dimension $\binom{d+m}{m}$.

### 5.4 Gaussian RBF Kernel

$$k(\mathbf{x}, \mathbf{z}) = \exp\left( -\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2} \right) = \exp\left( -\gamma \|\mathbf{x} - \mathbf{z}\|^2 \right)$$

where $\gamma = 1/(2\sigma^2)$. The RBF kernel is arguably the most widely used kernel in practice because:

- It corresponds to an **infinite-dimensional** feature space, so it can represent any continuous decision boundary.
- It is **local**: $k(\mathbf{x}, \mathbf{z}) \approx 0$ when $\|\mathbf{x} - \mathbf{z}\|$ is large, so distant points have negligible influence.
- It is **translation-invariant**: $k(\mathbf{x}, \mathbf{z}) = k(\mathbf{x} - \mathbf{z}, 0)$.
- The bandwidth $\sigma$ (or $\gamma$) controls smoothness: large $\sigma$ → smoother boundary, small $\sigma$ → more wiggly.

Using the Taylor expansion $e^{t} = \sum_{k=0}^\infty \frac{t^k}{k!}$:

$$k(\mathbf{x}, \mathbf{z}) = e^{-\|\mathbf{x}\|^2/2\sigma^2} e^{-\|\mathbf{z}\|^2/2\sigma^2} \sum_{k=0}^\infty \frac{(\mathbf{x} \cdot \mathbf{z})^k}{\sigma^{2k} k!}$$

revealing that the feature space is the direct sum of all homogeneous polynomial spaces of every degree.

### 5.5 Sigmoid Kernel

$$k(\mathbf{x}, \mathbf{z}) = \tanh(\kappa \, \mathbf{x} \cdot \mathbf{z} + \theta)$$

This kernel is related to two-layer neural networks but is **not** positive definite for all parameter choices — it must be used with care.

### 5.6 Other Notable Kernels

| Kernel | Formula | Notes |
|--------|---------|-------|
| Laplacian | $\exp(-\gamma \|\mathbf{x} - \mathbf{z}\|_1)$ | Heavy-tailed alternative to RBF |
| $\chi^2$ | $\sum \frac{2x_i z_i}{x_i + z_i}$ | Popular in computer vision |
| String kernel | Varies (e.g., all common substrings) | For text, bioinformatics |
| Graph kernel | Varies (e.g., random walks) | For structured data |

---

## 6. The Gram Matrix

For a training set $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$, the **Gram matrix** (or kernel matrix) is

$$\mathbf{K} = \big[ k(\mathbf{x}_i, \mathbf{x}_j) \big]_{i,j=1}^n \in \mathbb{R}^{n \times n}$$

Properties of $\mathbf{K}$ for a valid kernel:

1. **Symmetric:** $K_{ij} = K_{ji}$.
2. **Positive semidefinite:** For any $\mathbf{c} \in \mathbb{R}^n$, $\mathbf{c}^T \mathbf{K} \mathbf{c} \geq 0$.

The Gram matrix encodes all pairwise similarities between training points. In the SVM dual, the entire optimization (and thus the learned model) depends on the data solely through $\mathbf{K}$. This has computational implications:
- The dual has $n$ variables.
- Computing $\mathbf{K}$ takes $O(n^2 d)$ time.
- Solving the QP takes between $O(n^2)$ and $O(n^3)$ depending on implementation.
- Prediction for a new point costs $O(n_{\text{SV}} \, d) \approx O(n d)$.

For large $n$, these costs motivate scalable approximations (e.g., Nystrom method, random Fourier features).

---

## 7. Designing Kernels from Simple Kernels

Given valid kernels $k_1, k_2$ on $\mathcal{X}$, the following are also valid kernels (closed under these operations):

1. **Sum:** $k(\mathbf{x}, \mathbf{z}) = k_1(\mathbf{x}, \mathbf{z}) + k_2(\mathbf{x}, \mathbf{z})$
2. **Product:** $k(\mathbf{x}, \mathbf{z}) = k_1(\mathbf{x}, \mathbf{z}) \cdot k_2(\mathbf{x}, \mathbf{z})$
3. **Scalar multiplication:** $k(\mathbf{x}, \mathbf{z}) = c \cdot k_1(\mathbf{x}, \mathbf{z})$ for $c \geq 0$
4. **Limits:** If $(k_n)$ is a sequence of kernels converging pointwise, $\lim_n k_n$ is a kernel.
5. **Feature transform:** $k(\mathbf{x}, \mathbf{z}) = f(\mathbf{x}) k_1(\mathbf{x}, \mathbf{z}) f(\mathbf{z})$ for any $f: \mathcal{X} \to \mathbb{R}$
6. **Exponential:** $k(\mathbf{x}, \mathbf{z}) = \exp(k_1(\mathbf{x}, \mathbf{z}))$

These rules allow building complex, domain-specific kernels from elementary building blocks.

---

## 8. Kernels Beyond Vector Data

The kernel trick extends to any domain $\mathcal{X}$ where we can define a positive definite similarity function:

- **Strings:** Subsequence kernels, spectrum kernels for DNA/protein sequences.
- **Graphs:** Random walk kernels, shortest-path kernels, Weisfeiler-Lehman kernels.
- **Images:** Pyramid match kernels, spatial pyramid kernels.
- **Sets and bags:** Set kernels, earth mover's distance kernels.
- **Probability distributions:** Bhattacharyya kernel, probability product kernel.

This generality is one of the great strengths of kernel methods: separate the **data representation** (kernel design) from the **learning algorithm** (SVM, kernel ridge regression, kernel PCA, etc.).

---

## 9. Summary: Why the Kernel Trick Is "Magical"

1. **Computational efficiency:** Evaluate $k(\mathbf{x}, \mathbf{z})$ in $O(d)$ even when $\dim(\mathcal{F})$ is exponential or infinite.
2. **Algorithmic modularity:** Any algorithm expressible in terms of inner products — SVM, ridge regression, PCA, k-means, canonical correlation analysis — is instantly generalized to the nonlinear case.
3. **Domain flexibility:** Kernels exist for strings, graphs, images, and more — the algorithm stays the same, only the similarity measure changes.
4. **Statistical soundness:** The kernel defines a **Reproducing Kernel Hilbert Space** (RKHS), the natural function space for the learning problem. This connects to regularization theory and the representer theorem (see `rkhs.md`).

The kernel trick is not merely a computational shortcut — it is a deep conceptual bridge between linear methods in high-dimensional spaces and nonlinear methods in the original input space, grounded in the theory of Reproducing Kernel Hilbert Spaces.
