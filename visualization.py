"""
visualization.py

Utilities to plot results from BFO runs.
Uses matplotlib. Does not enforce colors/styles (respecte la consigne).
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_saved_figures = []


def plot_fitness_history(history: List[float], title: str = "Fitness history"):
    plt.figure(figsize=(8, 4.5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(True)
    filename = os.path.join(OUTPUT_DIR, "fitness_history.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    _saved_figures.append(filename)
    print(f"Saved fitness history to {filename}")


def plot_trajectories_2d(positions: np.ndarray, best_pos: np.ndarray = None, title: str = "Positions (2D)"):
    """
    Simple scatter of final positions and highlight best.
    If you have trajectories, extend this function to plot lines.
    """
    positions = np.asarray(positions)
    plt.figure(figsize=(6, 6))
    plt.scatter(positions[:, 0], positions[:, 1], s=30)
    if best_pos is not None:
        plt.scatter([best_pos[0]], [best_pos[1]], s=80, marker="*", edgecolors='k')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.grid(True)
    filename = os.path.join(OUTPUT_DIR, "positions_2d.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    _saved_figures.append(filename)
    print(f"Saved positions plot to {filename}")


def save_plots():
    print("Generated plots:")
    for fname in _saved_figures:
        print(" -", fname)
