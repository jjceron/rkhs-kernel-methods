"""
Experiment 5: RKHS Visualizations.

Creates educational demonstrations explaining:
  - Feature maps (explicit vs implicit)
  - Kernel evaluations as inner products
  - Gram matrix structure
  - RKHS geometric intuition
  - Why kernels avoid explicit high-dimensional computations

Outputs:
  figures/rkhs/05_feature_map.png
  figures/rkhs/05_implicit_embedding.png
  figures/rkhs/05_kernel_inner_product.png
  figures/rkhs/05_rkhs_geometry.png
  figures/rkhs/05_rkhs_intuition_demo.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.datasets import make_circles

from rkhs_kernel_methods.rkhs import (
    demonstrate_feature_map,
    demonstrate_implicit_embedding,
    demonstrate_kernel_inner_product,
    visualize_rkhs_geometry,
    rkhs_intuition_demo,
)
from rkhs_kernel_methods.visualization import set_style
from rkhs_kernel_methods.utils import ensure_dir, set_random_seed, Timer


def main() -> None:
    set_style()
    set_random_seed(42)

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE, "figures", "rkhs")
    ensure_dir(out_dir)

    print("=" * 60)
    print("EXPERIMENT 5: RKHS Visualizations")
    print("=" * 60)

    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

    print("\n1. Feature Map Demonstration...")
    with Timer("feature_map"):
        fig = demonstrate_feature_map(
            X,
            map_type="polynomial",
            degree=2,
            save_path=os.path.join(out_dir, "05_feature_map.png"),
        )
        print(f"   Saved to {os.path.join(out_dir, '05_feature_map.png')}")

    print("\n2. Implicit Embedding (Gram Matrix)...")
    with Timer("implicit_embedding"):
        fig = demonstrate_implicit_embedding(
            X,
            kernel="rbf",
            gamma=1.0,
            save_path=os.path.join(out_dir, "05_implicit_embedding.png"),
        )
        print(f"   Saved to {os.path.join(out_dir, '05_implicit_embedding.png')}")

    print("\n3. Kernel = Inner Product Verification...")
    with Timer("kernel_inner_product"):
        fig = demonstrate_kernel_inner_product(
            n_points=50,
            save_path=os.path.join(out_dir, "05_kernel_inner_product.png"),
        )
        print(f"   Saved to {os.path.join(out_dir, '05_kernel_inner_product.png')}")

    print("\n4. RKHS Geometry...")
    with Timer("rkhs_geometry"):
        fig = visualize_rkhs_geometry(
            n_samples=20,
            save_path=os.path.join(out_dir, "05_rkhs_geometry.png"),
        )
        print(f"   Saved to {os.path.join(out_dir, '05_rkhs_geometry.png')}")

    print("\n5. RKHS Intuition Demo...")
    with Timer("rkhs_intuition"):
        fig = rkhs_intuition_demo(
            save_path=os.path.join(out_dir, "05_rkhs_intuition_demo.png"),
        )
        print(f"   Saved to {os.path.join(out_dir, '05_rkhs_intuition_demo.png')}")

    print(f"\nAll figures saved to {out_dir}")
    print("Experiment 5 complete.")


if __name__ == "__main__":
    main()
