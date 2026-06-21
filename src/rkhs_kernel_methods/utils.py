"""
General-purpose utility functions for the RKHS kernel methods project.

Includes directory creation, random seed setting, timing context
managers, and figure saving helpers.
"""

import os
import time
import contextlib


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Directory path.

    Returns
    -------
    path : str
        The same path (for chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, filepath: str, close: bool = True) -> None:
    """
    Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str
        Output file path.
    close : bool, default=True
        Close the figure after saving.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed across numpy and Python's random module.

    Parameters
    ----------
    seed : int, default=42
    """
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)


@contextlib.contextmanager
def Timer(name: str = "Block"):
    """
    Context manager for timing code blocks.

    Usage
    -----
    with Timer("Training"):
        model.fit(X, y)

    Parameters
    ----------
    name : str, default='Block'
        Label for the timed block.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] Elapsed: {elapsed:.3f}s")
