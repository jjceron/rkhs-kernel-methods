"""
Master script: generates all figures by running experiments 1-5.

Usage:
    python scripts/06_generate_all_figures.py

Orchestrates the complete figure generation pipeline including
experiments and report summary figures.
"""

import subprocess
import sys
import os
import time


def run_script(script_name: str) -> None:
    """Run a Python script from the scripts/ directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base, script_name)
    print(f"\n{'#' * 60}")
    print(f"# Running: {script_name}")
    print(f"{'#' * 60}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
        cwd=os.path.dirname(base),
    )
    if result.returncode != 0:
        print(f"WARNING: {script_name} returned exit code {result.returncode}")


def main() -> None:
    print("=" * 60)
    print("GENERATING ALL FIGURES")
    print("=" * 60)

    scripts = [
        "01_linear_svm.py",
        "02_linear_failure.py",
        "03_kernel_trick.py",
        "04_kernel_comparison.py",
        "05_rkhs_visualizations.py",
        "05b_generate_summary_figures.py",
    ]

    start = time.perf_counter()

    for script in scripts:
        run_script(script)

    elapsed = time.perf_counter() - start
    print(f"\n{'=' * 60}")
    print(f"All figures generated in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
