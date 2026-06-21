# Support Vector Machine Theory

## 1. Introduction

Support Vector Machines (SVMs) are among the most elegant and theoretically well-founded classification algorithms in machine learning. Introduced by Vapnik and collaborators (Boser et al., 1992; Cortes & Vapnik, 1995), SVMs seek to find a **hyperplane** that separates two classes with the **maximum margin**, yielding a classifier with strong generalization guarantees. This document develops SVMs from first principles.

---

## 2. Binary Classification Setup

We are given a training set of $n$ labeled examples:

$$\mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^n, \quad \mathbf{x}_i \in \mathbb{R}^d, \quad y_i \in \{-1, +1\}$$

The goal is to learn a decision function $f: \mathbb{R}^d \to \{-1, +1\}$ that correctly classifies unseen data. We consider **linear** decision functions of the form

$$f(\mathbf{x}) = \operatorname{sign}\big( \mathbf{w} \cdot \mathbf{x} + b \big)$$

where $\mathbf{w} \in \mathbb{R}^d$ is the weight vector and $b \in \mathbb{R}$ is the bias term. The set $\{ \mathbf{x} : \mathbf{w} \cdot \mathbf{x} + b = 0 \}$ defines the **separating hyperplane**.

---

## 3. Hyperplanes and Decision Boundaries

A hyperplane in $\mathbb{R}^d$ is an affine subspace of dimension $d-1$:

$$H(\mathbf{w}, b) = \{ \mathbf{x} \in \mathbb{R}^d : \mathbf{w} \cdot \mathbf{x} + b = 0 \}$$

The vector $\mathbf{w}$ is orthogonal to the hyperplane and determines its orientation; $b$ determines its offset from the origin. The signed distance from a point $\mathbf{x}$ to the hyperplane is

$$\frac{\mathbf{w} \cdot \mathbf{x} + b}{\|\mathbf{w}\|}$$

Points with $\mathbf{w} \cdot \mathbf{x} + b > 0$ lie on one side; points with $\mathbf{w} \cdot \mathbf{x} + b < 0$ lie on the other.

---

## 4. Functional and Geometric Margins

**Functional margin.** For a training example $(\mathbf{x}_i, y_i)$, the functional margin with respect to a hyperplane $(\mathbf{w}, b)$ is

$$\hat{\gamma}_i = y_i (\mathbf{w} \cdot \mathbf{x}_i + b)$$

If $\hat{\gamma}_i > 0$, the example is classified correctly. The functional margin of the whole dataset is $\hat{\gamma} = \min_i \hat{\gamma}_i$.

**Geometric margin.** The functional margin is not invariant to scaling of $(\mathbf{w}, b)$. The **geometric margin** normalizes by $\|\mathbf{w}\|$ and represents the Euclidean distance from $\mathbf{x}_i$ to the hyperplane:

$$\gamma_i = y_i \left( \frac{\mathbf{w}}{\|\mathbf{w}\|} \cdot \mathbf{x}_i + \frac{b}{\|\mathbf{w}\|} \right) = \frac{\hat{\gamma}_i}{\|\mathbf{w}\|}$$

The geometric margin of the dataset is $\gamma = \min_i \gamma_i$.

---

## 5. Maximum Margin Classification

If the data are **linearly separable**, there exist infinitely many hyperplanes that correctly classify the training data. The SVM selects the one that maximizes the geometric margin:

$$\max_{\mathbf{w}, b} \; \gamma \quad \text{s.t.} \quad y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq \gamma \|\mathbf{w}\|,\; \forall i$$

Without loss of generality, we rescale $(\mathbf{w}, b)$ so that the functional margin equals 1 for the points closest to the hyperplane. This yields the canonical **primal optimization problem**:

$$\boxed{ \min_{\mathbf{w}, b} \; \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1,\; i = 1, \dots, n }$$

Minimizing $\frac{1}{2}\|\mathbf{w}\|^2$ is equivalent to maximizing $1/\|\mathbf{w}\|$, i.e., maximizing the geometric margin. The constraints enforce that every training point lies at least distance $1/\|\mathbf{w}\|$ from the hyperplane, on the correct side.

### Why quadratic? Why $\frac{1}{2}\|\mathbf{w}\|^2$?

- The objective is **convex** and **differentiable**, so we can use tools from convex optimization.
- The factor $1/2$ is for algebraic convenience (cancels when differentiating).
- The problem is a **quadratic program** (QP) — well-studied and efficiently solvable.

---

## 6. Support Vectors

The constraints are **active** at points closest to the hyperplane:

$$y_i (\mathbf{w} \cdot \mathbf{x}_i + b) = 1$$

These points are called **support vectors** because they "support" the hyperplane: if we remove all other points, the solution does not change. Support vectors are the **only** data points that influence the final decision boundary — a remarkable property that underlies SVM sparsity and generalization.

Let $SV \subseteq \{1, \dots, n\}$ denote the index set of support vectors. The margin is

$$\gamma = \frac{1}{\|\mathbf{w}\|}$$

and the optimal $\mathbf{w}$ lies in the span of the support vectors (as we shall see via the dual).

---

## 7. Soft Margin SVM (Linearly Non-Separable Case)

Real data are rarely perfectly separable. We introduce **slack variables** $\xi_i \geq 0$ that allow points to violate the margin:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The primal becomes

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \; \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i,\; \xi_i \geq 0$$

The hyperparameter $C > 0$ controls the trade-off between maximizing the margin and minimizing classification error:
- **Small $C$**: wide margin, more tolerance for misclassification.
- **Large $C$**: narrower margin, less tolerance; approaches hard-margin SVM.

---

## 8. Lagrangian and Dual Formulation

### 8.1 Lagrangian of the Hard-Margin Problem

Introduce Lagrange multipliers $\alpha_i \geq 0$ (one per constraint). The Lagrangian is

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \big[ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \big]$$

The **primal problem** is $\min_{\mathbf{w}, b} \max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}$; the **dual problem** is $\max_{\boldsymbol{\alpha} \geq 0} \min_{\mathbf{w}, b} \mathcal{L}$. By strong duality (Slater's condition holds), both have the same optimum.

### 8.2 Eliminating the Primal Variables

Set derivatives to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 \quad \Rightarrow \quad \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^n \alpha_i y_i = 0$$

Substituting back yields the **dual problem**:

$$\boxed{ \max_{\boldsymbol{\alpha}} \; \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j) }$$

subject to $\alpha_i \geq 0$ and $\sum_i \alpha_i y_i = 0$.

### 8.3 Karush-Kuhn-Tucker (KKT) Conditions

For the optimal solution:

1. **Primal feasibility**: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$
2. **Dual feasibility**: $\alpha_i \geq 0$
3. **Complementary slackness**: $\alpha_i \big[ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \big] = 0$
4. **Stationarity**: derivatives equal zero

Complementary slackness implies:
- If $\alpha_i > 0$, then $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$ (point is a support vector).
- If $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) > 1$, then $\alpha_i = 0$ (point lies strictly outside the margin zone).

### 8.4 Decision Function in Dual Form

Once optimal $\boldsymbol{\alpha}^\star$ is found, we compute

$$\mathbf{w}^\star = \sum_{i=1}^n \alpha_i^\star y_i \mathbf{x}_i$$

and $b^\star = y_k - \mathbf{w}^\star \cdot \mathbf{x}_k$ for any support vector $k$ (with $\alpha_k > 0$). The decision function becomes

$$f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i=1}^n \alpha_i^\star y_i (\mathbf{x}_i \cdot \mathbf{x}) + b^\star \right)$$

Notice that only support vectors (those with $\alpha_i > 0$) contribute to the sum — the decision function is **sparse**.

### 8.5 Dual of the Soft-Margin Problem

For the soft-margin case, the dual is identical except for the bound on $\alpha_i$:

$$\max_{\boldsymbol{\alpha}} \; \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$.

The KKT conditions now include $\alpha_i \leq C$, and points with $\alpha_i = C$ are on the wrong side of the margin (misclassified or within the margin).

---

## 9. Why SVM Generalizes Well

### 9.1 Structural Risk Minimization

Vapnik's theory of **structural risk minimization (SRM)** bounds the expected risk (test error) by the empirical risk plus a capacity term. By maximizing the margin, SVM minimizes an upper bound on the VC dimension:

$$\text{VC dimension of margin hyperplanes} \leq \min\left( \left\lceil \frac{R^2}{\gamma^2} \right\rceil, \, d \right) + 1$$

where $R$ is the radius of the smallest sphere enclosing the data and $\gamma$ is the margin. Maximizing $\gamma$ directly reduces the VC dimension, improving generalization.

### 9.2 Sparsity

Because only support vectors ($\alpha_i > 0$) contribute to the decision function, the solution is sparse. The effective model complexity is controlled by the number of support vectors rather than the input dimension — this is key to preventing overfitting in high-dimensional settings.

### 9.3 Convexity

The dual problem is a **convex quadratic program** with a unique maximum (in $\boldsymbol{\alpha}$). There are no local minima — any local optimum is global. This makes SVM reliably solvable.

---

## 10. From Linear to Nonlinear: Preview

The dual formulation reveals an essential fact: the optimization problem and the decision function depend on the data **only through inner products** $\mathbf{x}_i \cdot \mathbf{x}_j$ and $\mathbf{x}_i \cdot \mathbf{x}$. This observation is the gateway to the **kernel trick**, where we replace inner products with kernel evaluations:

$$\mathbf{x}_i \cdot \mathbf{x}_j \;\longrightarrow\; k(\mathbf{x}_i, \mathbf{x}_j)$$

This allows SVMs to learn nonlinear decision boundaries without ever constructing explicit high-dimensional feature maps — the subject of the next document.

---

## 11. Summary

| Concept | Expression |
|---------|-----------|
| Primal (hard margin) | $\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2$ s.t. $y_i(\mathbf{w}\cdot\mathbf{x}_i+b) \ge 1$ |
| Primal (soft margin) | $\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$ s.t. $y_i(\mathbf{w}\cdot\mathbf{x}_i+b) \ge 1-\xi_i$ |
| Dual (both) | $\max_{\boldsymbol{\alpha}} \sum\alpha_i - \frac{1}{2}\sum\alpha_i\alpha_j y_i y_j (\mathbf{x}_i\cdot\mathbf{x}_j)$ |
| Decision | $f(\mathbf{x}) = \operatorname{sign}\!\big(\sum\alpha_i y_i (\mathbf{x}_i\cdot\mathbf{x}) + b\big)$ |
| $\mathbf{w}$ in dual | $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$ |
| Support vectors | $\{i : \alpha_i > 0\}$ |
| Margin | $\gamma = 1/\|\mathbf{w}\|$ |

The SVM framework elegantly combines convex optimization, duality theory, and generalization bounds into a principled classifier. Its reliance on inner products makes it the perfect foundation for kernel methods.
