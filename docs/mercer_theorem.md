# Mercer's Theorem for Beginners

## 1. Introduction: What Problem Does Mercer's Theorem Solve?

Suppose you have designed a function $k(\mathbf{x}, \mathbf{z})$ that measures similarity between data points. You want to use it in an SVM or another kernel method. The central question is:

> **Is $k$ a valid kernel? Does there exist a feature space $\mathcal{F}$ and a feature map $\varphi$ such that $k(\mathbf{x}, \mathbf{z}) = \langle \varphi(\mathbf{x}), \varphi(\mathbf{z}) \rangle$?**

Mercer's theorem gives a definitive answer for a large and important class of kernels on compact domains. It not only certifies kernel validity but also provides an explicit **eigen-expansion** that reveals the structure of the associated feature space and RKHS.

This document develops Mercer's theorem with maximum intuition and minimal technicality.

---

## 2. Positive Semidefinite Kernels

### 2.1 Definition

A kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is **positive semidefinite (PSD)** if for any finite set of points $\\{ \mathbf{x}_1, \dots, \mathbf{x}_n \\} \subseteq \mathcal{X}$ and any real coefficients $c_1, \dots, c_n$, the following quadratic form is nonnegative:

$$\sum_{i=1}^n \sum_{j=1}^n c_i c_j \\, k(\mathbf{x}_i, \mathbf{x}_j) \geq 0$$

In matrix terms: the **Gram matrix** $\mathbf{K}$ with entries $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ must be positive semidefinite for every choice of points.

### 2.2 What Does "Positive Semidefinite" Mean Intuitively?

Think of the Gram matrix as a table of pairwise similarities. The PSD condition ensures that these similarities come from an **inner product structure** — objects are "similar" in a way that is geometrically consistent.

- **Analogy:** In Euclidean space, if three points are mutually similar, the third cannot simultaneously be dissimilar to both. The PSD condition enforces this kind of **triangle-like consistency** on similarity judgments.
- **Another view:** A PSD Gram matrix can always be factorized as $\mathbf{K} = \mathbf{\Phi}^T \mathbf{\Phi}$ where $\mathbf{\Phi}$ is an $m \times n$ matrix whose columns are the feature vectors of the data points. The entries of $\mathbf{\Phi}$ are the feature representations.

- **A non-example:** $k(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x} \cdot \mathbf{z})$ is NOT PSD for $n > 2$ — it can produce Gram matrices with negative eigenvalues.

---

## 3. Preamble: Integral Operators and Eigenvalues

Mercer's theorem concerns a **continuous** analogue of the Gram matrix. For a compact metric space $\mathcal{X}$ and a finite measure $\mu$ on $\mathcal{X}$, define the **integral operator** $T_k: L^2(\mathcal{X}, \mu) \to L^2(\mathcal{X}, \mu)$ by

$$(T_k f)(\mathbf{x}) = \int_{\mathcal{X}} k(\mathbf{x}, \mathbf{z}) f(\mathbf{z}) \\, d\mu(\mathbf{z})$$

If $k$ is continuous and PSD, then $T_k$ is:
- **Compact** (maps bounded sets to relatively compact sets).
- **Self-adjoint**: $\langle T_k f, g \rangle = \langle f, T_k g \rangle$.
- **Positive semidefinite**: $\langle T_k f, f \rangle \geq 0$ for all $f$.

The **spectral theorem** for compact self-adjoint operators then guarantees that $T_k$ has a discrete spectrum:

$$T_k \phi_i = \lambda_i \phi_i, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq 0, \quad \lambda_i \to 0$$

where $\{ \phi_i \}$ is an orthonormal basis of $L^2(\mathcal{X}, \mu)$ consisting of eigenfunctions, and $\lambda_i \geq 0$ (nonnegativity follows from PSD).

---

## 4. Mercer's Theorem: Statement

**Theorem (Mercer, 1909).** Let $\mathcal{X}$ be a compact metric space and let $\mu$ be a finite Borel measure with support $\mathcal{X}$. Let $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a continuous, symmetric, positive semidefinite kernel. Then there exist nonnegative eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$ and continuous eigenfunctions $\phi_i \in L^2(\mathcal{X}, \mu)$ such that

$$\boxed{ k(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^\infty \lambda_i \\, \phi_i(\mathbf{x}) \\, \phi_i(\mathbf{z}) }$$

The series converges **absolutely** and **uniformly** on $\mathcal{X} \times \mathcal{X}$.

### Breaking Down the Statement

| Component | Meaning |
|-----------|---------|
| "Continuous, symmetric, PSD" | The kernel is well-behaved and geometrically consistent |
| $\lambda_i \geq 0$ | Eigenvalues are nonnegative — the PSD property at work |
| $\phi_i$ are orthonormal in $L^2$ | $\int \phi_i(\mathbf{x}) \phi_j(\mathbf{x}) d\mu(\mathbf{x}) = \delta_{ij}$ |
| Uniform convergence | The expansion converges at the same rate everywhere on $\mathcal{X} \times \mathcal{X}$ |
| $\lambda_i \to 0$ | Eigenvalues decay to zero (consequence of compactness) |

---

## 5. Connection to Feature Maps

Mercer's expansion directly provides a feature map. Define

$$\varphi(\mathbf{x}) = \left( \sqrt{\lambda_1} \\, \phi_1(\mathbf{x}),\; \sqrt{\lambda_2} \\, \phi_2(\mathbf{x}),\; \dots \right) \in \ell^2$$

Then

$$k(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^\infty \lambda_i \phi_i(\mathbf{x}) \phi_i(\mathbf{z}) = \langle \varphi(\mathbf{x}), \varphi(\mathbf{z}) \rangle_{\ell^2}$$

This feature map is infinite-dimensional in general (the sum runs to $\infty$). But the eigenvalues $\lambda_i$ decay, and if they decay rapidly, the effective dimensionality — the number of features needed for a good approximation — is modest.

The Mercer expansion decomposes the kernel into **orthogonal spectral components**. Each eigenfunction $\phi_i$ captures a "mode of variation" in the data, and $\lambda_i$ measures the importance (variance) of that mode. This is the kernel analogue of PCA.

---

## 6. Why Mercer's Theorem Guarantees a Valid RKHS

The Mercer expansion provides the canonical feature map $\varphi(\mathbf{x}) = (\sqrt{\lambda_i} \phi_i(\mathbf{x}))_{i=1}^\infty \in \ell^2$. The RKHS $\mathcal{H}_k$ can be characterized as

$$\mathcal{H}_k = \left\\{ f = \sum_{i=1}^\infty a_i \phi_i \;\Bigg|\; \sum_{i=1}^\infty \frac{a_i^2}{\lambda_i} < \infty \right\\}$$

with inner product

$$\langle f, g \rangle_{\mathcal{H}_k} = \sum_{i=1}^\infty \frac{a_i b_i}{\lambda_i}$$

where $f = \sum a_i \phi_i$ and $g = \sum b_i \phi_i$.

**Key insight:** The RKHS consists of functions whose expansion coefficients $a_i$ decay faster than $\sqrt{\lambda_i}$. Functions in $\mathcal{H}_k$ are **smoother** than generic $L^2$ functions — the kernel enforces a smoothness constraint through the eigenvalue decay.

The reproducing property can be verified:

$$f(\mathbf{x}) = \sum_i a_i \phi_i(\mathbf{x}) = \sum_i \frac{a_i}{\lambda_i} \lambda_i \phi_i(\mathbf{x}) = \left\langle \sum_i a_i \phi_i, \; \sum_j \lambda_j \phi_j(\mathbf{x}) \phi_j \right\rangle_{\mathcal{H}_k}$$

$$= \langle f, k(\cdot, \mathbf{x}) \rangle_{\mathcal{H}_k}$$

---

## 7. How to Check if a Kernel Is Valid in Practice

### 7.1 Direct PSD Test

For a proposed kernel $k$:
1. Draw a random sample of $n$ points from your domain.
2. Compute the $n \times n$ Gram matrix $\mathbf{K}$.
3. Check its eigenvalues: if any are significantly negative (beyond numerical error), $k$ is likely not PSD.

This is an **empirical necessary condition** — if it fails, the kernel is invalid. If it passes for many random samples, the kernel is likely valid.

### 7.2 Analytical Tests

- **Expansion as inner product:** If you can exhibit $\varphi$ explicitly, $k$ is valid trivially.
- **Closure properties:** If $k$ is built from valid kernels using known closure operations (sum, product, scalar multiplication, limit, exponential), it is valid.
- **Conditional positive definiteness:** Some kernels (e.g., negative distance $k(\mathbf{x}, \mathbf{z}) = -\|\mathbf{x} - \mathbf{z}\|$) are not PSD but are **conditionally PSD** — they can be used with appropriate constraints.

### 7.3 Breakdown of Mercer's Conditions

If $\mathcal{X}$ is not compact, or $k$ is not continuous, Mercer's theorem does not apply directly. However, the connection between PSD kernels and RKHS (Moore-Aronszajn) holds for arbitrary sets $\mathcal{X}$.

---

## 8. Spectral Decay and Generalization

The rate at which $\lambda_i \to 0$ dictates the **capacity** of the RKHS:

- **Fast decay** (e.g., $\lambda_i \sim e^{-i}$): The RKHS is a "small" space of very smooth functions. Learning is easy (low sample complexity) but the model may be biased if the true function is not smooth enough.
- **Slow decay** (e.g., $\lambda_i \sim i^{-\alpha}$ for small $\alpha$): The RKHS is "large" — it can represent more complex functions, but learning requires more data.

This spectral perspective unifies kernel choice with generalization analysis. Modern theories of kernel learning (e.g., Caponnetto & De Vito, 2007) derive optimal learning rates in terms of the eigenvalue decay rate.

---

## 9. Mercer Expansions of Common Kernels

### 9.1 Linear Kernel on $[-1, 1]$

$$k(x, z) = xz$$

With Lebesgue measure on $[-1, 1]$, the eigenfunctions are Legendre polynomials scaled appropriately. The expansion is finite (one term effectively), since the feature map is finite-dimensional.

### 9.2 Polynomial Kernel

$$k(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^m$$

The Mercer expansion has finitely many nonzero eigenvalues (the feature space is finite-dimensional). The number of nonzero $\lambda_i$ equals $\binom{d+m-1}{m}$, the dimension of the homogeneous polynomial space of degree $m$ in $d$ variables.

### 9.3 Gaussian RBF Kernel on $\mathbb{R}^d$

With Gaussian measure $d\mu(\mathbf{x}) = (2\pi)^{-d/2} e^{-\|\mathbf{x}\|^2/2} d\mathbf{x}$, the eigenfunctions of the RBF kernel are **Hermite polynomials**, and the eigenvalues are

$$\lambda_{\mathbf{k}} = \sqrt{\frac{2a}{A}} \left( 1 - \\, \frac{1}{A} \right)^{|\mathbf{k}|}$$

where $a$ is the kernel bandwidth parameter, $A = a + b + c$ involves the measure bandwidth, and $\mathbf{k}$ is a multi-index. The eigenvalues decay geometrically, confirming extreme smoothness.

In the infinite-dimensional limit ($\mathbb{R}^d$ with Lebesgue measure), the RBF kernel has a **continuous spectrum** — Mercer's theorem does not directly apply, but the connection to RKHS theory still holds via Moore-Aronszajn.

---

## 10. Summary: What Mercer's Theorem Tells Us

1. **Certification:** A continuous PSD kernel on a compact domain is guaranteed to have a feature map — it is a valid kernel.

2. **Construction:** The feature map is given explicitly by $\varphi(\mathbf{x}) = (\sqrt{\lambda_i} \phi_i(\mathbf{x}))_{i=1}^\infty$.

3. **Structure:** The kernel is a superposition of rank-one PSD components $\lambda_i \phi_i(\mathbf{x}) \phi_i(\mathbf{z})$, each representing an independent "dimension" of similarity with importance $\lambda_i$.

4. **Smoothness:** The RKHS norm filters functions by their expansion coefficients relative to $\sqrt{\lambda_i}$ — this penalizes high-frequency (small $\lambda_i$) components.

5. **Approximation:** Truncating the expansion at the $N$ largest eigenvalues gives the best rank-$N$ approximation of the kernel (in the sense of the integral operator norm), a property useful for scalable kernel methods (Nystrom approximation).

Mercer's theorem is the bridge between **linear algebra** (Gram matrices, eigenvalues) and **functional analysis** (integral operators, function spaces). It is the mathematical reason why we can trust kernels as implicit feature maps.
