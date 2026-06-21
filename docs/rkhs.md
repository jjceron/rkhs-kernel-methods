# Reproducing Kernel Hilbert Spaces

## 1. Introduction

Reproducing Kernel Hilbert Spaces (RKHS) are the mathematical foundation of kernel methods. They provide a rigorous framework for understanding why the kernel trick works, how kernels define function spaces, and why learning in these spaces is well-posed. An RKHS is a Hilbert space of functions where **point evaluation is a continuous (bounded) linear functional** — a property that turns out to be profoundly useful for optimization and learning.

This document builds the RKHS concept from first principles, connecting it to SVMs via the representer theorem.

---

## 2. Hilbert Spaces: A Quick Review

A **Hilbert space** $\mathcal{H}$ is a complete inner product space. For our purposes, this means:

- $\mathcal{H}$ is a real vector space (or complex, but we consider real).
- It has an inner product $\langle \cdot, \cdot \rangle_{\mathcal{H}}: \mathcal{H} \times \mathcal{H} \to \mathbb{R}$ that is symmetric, bilinear, and positive definite: $\langle f, f \rangle \geq 0$, with equality iff $f = 0$.
- The inner product induces a norm $\|f\|_{\mathcal{H}} = \sqrt{\langle f, f \rangle_{\mathcal{H}}}$.
- $\mathcal{H}$ is **complete**: every Cauchy sequence converges to a limit in $\mathcal{H}$.

**Examples of Hilbert spaces:**
- $\mathbb{R}^n$ with the Euclidean inner product $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_i u_i v_i$.
- $L^2(\Omega) = \{ f: \Omega \to \mathbb{R} : \int_\Omega f(x)^2 dx < \infty \}$ with $\langle f, g \rangle = \int f(x) g(x) dx$.
- $\ell^2$, the space of square-summable sequences.

---

## 3. Function Spaces as Hilbert Spaces

In machine learning, we are interested in Hilbert spaces whose elements are **functions** from an input set $\mathcal{X}$ to $\mathbb{R}$, not just finite-dimensional vectors.

A function space $\mathcal{H} \subseteq \\{ f : \mathcal{X} \to \mathbb{R} \\}$ is a Hilbert space if there exists an inner product $\langle \cdot, \cdot \rangle_{\mathcal{H}}$ under which $\mathcal{H}$ is complete.

However, not every Hilbert space of functions has the "nice" property that **evaluation at a point** is continuous. For the theory of kernel methods, this property is essential.

---

## 4. The Evaluation Functional

For each $x \in \mathcal{X}$, define the **evaluation functional** $L_x: \mathcal{H} \to \mathbb{R}$ by

$$L_x(f) = f(x)$$

That is, $L_x$ maps a function $f$ to its value at the point $x$.

In a general Hilbert space of functions, $L_x$ may not be continuous: small changes in $f$ (in the Hilbert norm) could produce large changes in $f(x)$. For $\mathcal{H}$ to be an RKHS, we require $L_x$ to be **bounded** (equivalently, continuous) for every $x$.

---

## 5. Definition of an RKHS

A Hilbert space $\mathcal{H}$ of functions $f: \mathcal{X} \to \mathbb{R}$ is a **Reproducing Kernel Hilbert Space** if for every $x \in \mathcal{X}$, the evaluation functional $L_x: f \mapsto f(x)$ is bounded, i.e.,

$$\exists M_x > 0 \text{ such that } |L_x(f)| = |f(x)| \leq M_x \\|f\\|_{\mathcal{H}} \quad \forall f \in \mathcal{H}$$

This property has far-reaching consequences.

---

## 6. The Reproducing Property

Since $L_x$ is a bounded linear functional, by the **Riesz representation theorem** there exists a unique element $k_x \in \mathcal{H}$ such that

$$L_x(f) = \langle f, k_x \rangle_{\mathcal{H}} \quad \forall f \in \mathcal{H}$$

In other words:

$$\boxed{ f(x) = \langle f, k(\\\cdot, x) \\rangle_{\mathcal{H}} }$$

where $k_x(\cdot) = k(\cdot, x)$. This is the **reproducing property** — the function value at any point can be "reproduced" as the inner product with a kernel function centered at that point.

### The Kernel Function

Define $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ by

$$k(x, y) = \langle k_x, k_y \rangle_{\mathcal{H}} = \langle k(\cdot, x), k(\cdot, y) \rangle_{\mathcal{H}}$$

Applying the reproducing property to the function $k_y$ evaluated at $x$:

$$k_y(x) = k(x, y) = \langle k_y, k_x \rangle_{\mathcal{H}} = \langle k_x, k_y \rangle_{\mathcal{H}} = k(y, x)$$

so $k(x, y) = k(y, x)$; the kernel is symmetric.

Moreover, $k_x$ is precisely $k(\cdot, x) \in \mathcal{H}$ — the function of the first argument with the second fixed at $x$. This is the **canonical feature map**:

$$\varphi(x) = k(\cdot, x)$$

and then

$$k(x, y) = \langle \varphi(x), \varphi(y) \rangle_{\mathcal{H}}$$

Every RKHS comes with a unique **reproducing kernel** $k$, and every positive definite kernel $k$ defines a unique RKHS. This is the fundamental one-to-one correspondence.

---

## 7. Every Positive Definite Kernel Defines a Unique RKHS

**Theorem (Moore-Aronszajn).** Let $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a positive definite kernel (i.e., for any $\{x_i\}_{i=1}^n \subseteq \mathcal{X}$ and any $c_i \in \mathbb{R}$, $\sum_{i,j} c_i c_j k(x_i, x_j) \geq 0$). Then there exists a unique RKHS $\mathcal{H}_k$ with reproducing kernel $k$.

**Sketch of construction:**

1. Start with the pre-Hilbert space $\mathcal{H}_0 = \operatorname{span}\\{ k(\cdot, x) : x \in \mathcal{X} \\}$, i.e., all finite linear combinations:

   $$f = \sum_{i=1}^m \alpha_i k(\cdot, x_i)$$

2. Define the inner product for $f = \sum_i \alpha_i k(\cdot, x_i)$ and $g = \sum_j \beta_j k(\cdot, y_j)$ as

   $$\langle f, g \rangle_{\mathcal{H}} = \sum_{i=1}^m \sum_{j=1}^n \alpha_i \beta_j \\, k(x_i, y_j)$$

   Positive definiteness of $k$ guarantees this is a well-defined inner product.

3. The reproducing property holds by construction: for $f = \sum_i \alpha_i k(\cdot, x_i)$,

   $$\langle f, k(\cdot, x) \rangle = \sum_i \alpha_i k(x_i, x) = f(x)$$

4. Complete $\mathcal{H}_0$ to obtain $\mathcal{H}_k$.

The functions $k(\cdot, x)$ thus act as "basis atoms" that span the whole space.

---

## 8. Intuition: Kernel Functions as "Atoms"

Think of the kernel $k(\cdot, x)$ as a **bump function** centered at $x$. The RKHS $\mathcal{H}_k$ is the space of all weighted superpositions of these bumps:

$$f = \sum_{i=1}^\infty \alpha_i \\, k(\cdot, x_i)$$

The inner product aligns with the geometry induced by $k$: two functions are close in $\mathcal{H}_k$ norm if they are similar on training points, as measured by $k$.

The norm $\|f\|_{\mathcal{H}_k}$ measures the **smoothness** or **complexity** of $f$ with respect to the kernel. Functions that oscillate rapidly in the metric induced by $k$ have large norm; functions that are smooth (i.e., representable with few kernel atoms) have small norm. This makes $\|f\|_{\mathcal{H}}$ a natural regularizer:

$$\min_{f \in \mathcal{H}_k} \; \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \\|f\\|_{\mathcal{H}}^2$$

---

## 9. Connection to SVM: The Representer Theorem

### 9.1 Statement

**Theorem (Kimeldorf & Wahba, 1971; Scholkopf et al., 2001).** Let $\mathcal{H}_k$ be an RKHS with kernel $k$, let $\lambda > 0$, and let $L$ be any loss function depending on $f$ only through $f(x_1), \dots, f(x_n)$. Then any minimizer of

$$\min_{f \in \mathcal{H}_k} \; \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \\|f\\|_{\mathcal{H}}^2$$

admits a representation of the form

$$f^\star = \sum_{i=1}^n \alpha_i \\, k(\cdot, x_i)$$

where $\alpha_i \in \mathbb{R}$.

### 9.2 Significance

The representer theorem is profound: although $\mathcal{H}_k$ may be **infinite-dimensional**, the optimal solution lies in a subspace of dimension at most $n$ (the number of training points). This makes the infinite-dimensional problem computationally tractable.

### 9.3 Application to SVM

The SVM optimization can be recast in RKHS terms:

$$\min_{f \in \mathcal{H}_k} \; \frac{1}{2}\\|f\\|_{\mathcal{H}}^2 + C \sum_{i=1}^n \max(0, 1 - y_i f(x_i))$$

By the representer theorem, $f^\star = \sum_i \alpha_i k(\cdot, x_i)$, and the problem reduces to the familiar SVM dual in $\boldsymbol{\alpha}$.

### 9.4 Proof Sketch

Decompose any $f \in \mathcal{H}_k$ as $f = f_\parallel + f_\perp$, where $f_\parallel \in \operatorname{span}\\{k(\cdot, x_i)\\}$ and $f_\perp$ is orthogonal to this span. Then:
- $f_\perp(x_i) = \langle f_\perp, k(\cdot, x_i) \rangle = 0$ for all $i$, so $f_\perp$ does not affect the loss term.
- But $\|f\|^2 = \|f_\parallel\|^2 + \|f_\perp\|^2 \geq \|f_\parallel\|^2$.

Thus $f_\perp$ only increases the objective — the optimal $f$ must have $f_\perp = 0$, hence lies in the span of the kernel functions centered at the training points.

---

## 10. Norm and Regularization in RKHS

The RKHS norm measures function complexity. For the Gaussian RBF kernel:

$$\|f\|_{\mathcal{H}}^2 = \int_{\mathbb{R}^d} \sum_{m=0}^\infty \frac{\sigma^{2m}}{m! \\, 2^m} \\, (\mathcal{D}^m f)^2 \\, d\rho(x)$$

where $\mathcal{D}^m$ is the $m$-th order derivative operator and $\rho$ is a Gaussian measure. This reveals a precise connection: smoothing with an RBF kernel penalizes **all** derivatives of $f$ — higher-order derivatives are penalized more when $\sigma$ is large, and less when $\sigma$ is small. This is why $\sigma$ controls the smoothness of the learned function.

---

## 11. Why RKHS Matters for Machine Learning

| Property | What it gives us |
|----------|-----------------|
| Reproducing property | Point evaluation through inner products; enables kernel trick |
| Moore-Aronszajn theorem | Every PSD kernel ↔ a unique RKHS; justifies kernel design |
| Representer theorem | Optimal solution lies in finite-dimensional span of training kernels |
| RKHS norm | Natural complexity measure / regularizer |
| Completeness | Limits of Cauchy sequences stay in $\mathcal{H}$; convergence is well-behaved |
| Inner product structure | Enables orthogonal projections, subspaces, and least-squares formulations |

The RKHS framework unifies:
- **Kernel design** (choose $k$ → choose the hypothesis space $\mathcal{H}_k$)
- **Regularization** (penalize $\|f\|_{\mathcal{H}}$ → control complexity)
- **Optimization** (representer theorem → finite-dimensional problem)
- **Generalization** (bounds in terms of RKHS norm and kernel spectrum)

---

## 12. Examples of RKHS

### 12.1 Linear Kernel on $\mathbb{R}^d$

$$k(\mathbf{x}, \mathbf{y}) = \mathbf{x} \cdot \mathbf{y}$$

The RKHS is $\{ f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} : \mathbf{w} \in \mathbb{R}^d \}$, i.e., the space of linear functions, with $\|f\|_{\mathcal{H}} = \|\mathbf{w}\|$.

### 12.2 Polynomial Kernel

$$k(\mathbf{x}, \mathbf{y}) = (\mathbf{x} \cdot \mathbf{y} + 1)^m$$

The RKHS consists of all polynomials of degree $\leq m$, with a norm that penalizes the coefficients of higher-degree monomials.

### 12.3 Gaussian RBF Kernel

$$k(\mathbf{x}, \mathbf{y}) = \exp\left( -\frac{\\|\mathbf{x} - \mathbf{y}\\|^2}{2\sigma^2} \right)$$

The RKHS is an infinite-dimensional space of smooth functions. It can be described explicitly as a weighted Sobolev space:

$$\mathcal{H}_k = \left\\{ f \in L^2(\mathbb{R}^d) : \int \| \hat{f}(\omega) \|^2 \\, e^{\sigma^2 \|\omega\|^2 / 2} \\, d\omega < \infty \right\\}$$

where $\hat{f}$ is the Fourier transform of $f$. The condition demands that the Fourier transform of $f$ decay faster than a Gaussian — i.e., $f$ must be infinitely differentiable and extremely smooth.

---

## 13. Summary Equation Box

| Concept | Equation |
|---------|----------|
| Recovering a kernel from an RKHS | $k(x, y) = \langle k(\cdot, x), k(\cdot, y) \rangle_{\mathcal{H}}$ |
| Reproducing property | $f(x) = \langle f, k(\cdot, x) \rangle_{\mathcal{H}}$ |
| Canonical feature map | $\varphi(x) = k(\cdot, x) \in \mathcal{H}_k$ |
| Representer theorem solution | $f^\star = \sum_{i=1}^n \alpha_i k(\cdot, x_i)$ |
| Regularized risk | $\min_{f \in \mathcal{H}_k} \frac{1}{n}\sum_i L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}}^2$ |

The RKHS is the "universe" in which kernel methods live. Understanding it reveals why kernels are not just a computational trick but a deep, principled framework for nonparametric function estimation.
