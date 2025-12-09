"""
objective_functions.py

Collection of benchmark objective functions for testing BFO.

Each function signature:
    func(x: np.ndarray) -> float

Also helper to get bounds for a function.
"""

import numpy as np
from typing import Tuple


def sphere(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2). Global minimum at 0."""
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: multimodal, global minimum at 0."""
    A = 10.0
    n = x.size
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function (classic valley). Minimum at [1,...,1]."""
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def ackley(x: np.ndarray) -> float:
    """Ackley function. Global minimum at 0."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.size
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e)


def get_default_bounds(name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower_bounds, upper_bounds) arrays for given function name."""
    if name.lower() == "sphere":
        return -5.12 * np.ones(dim), 5.12 * np.ones(dim)
    if name.lower() == "rastrigin":
        return -5.12 * np.ones(dim), 5.12 * np.ones(dim)
    if name.lower() == "rosenbrock":
        return -2.048 * np.ones(dim), 2.048 * np.ones(dim)
    if name.lower() == "ackley":
        return -32.768 * np.ones(dim), 32.768 * np.ones(dim)
    # default generic bounds
    return -5.0 * np.ones(dim), 5.0 * np.ones(dim)


# simple registry
FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
}
