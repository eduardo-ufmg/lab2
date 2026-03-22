import collections
from collections.abc import Callable

import numpy as np


def create_filter(
    alpha: float, median_N: int = 3, initial_state: float = 0.0
) -> Callable[[float], float]:
    if not (0.0 <= alpha < 1.0):
        raise ValueError("Pole alpha must strictly fall within the interval [0, 1).")
    if median_N < 1 or median_N % 2 == 0:
        raise ValueError("median_N must be a strictly positive odd integer.")

    x_buffer = collections.deque([initial_state] * median_N, maxlen=median_N)
    feedforward_coeff = 1.0 - alpha
    y_prev = initial_state  # Python 3 scalar state

    def filter_func(x: float) -> float:
        nonlocal y_prev  # Declare nonlocal scope

        x_buffer.append(x)

        # Dynamic median operation
        x_med = sorted(x_buffer)[median_N // 2]

        # IIR difference equation
        y = alpha * y_prev + feedforward_coeff * x_med
        y_prev = y

        return y

    return filter_func


def test_filter(
    alpha: float, median_N: int, y: np.ndarray, initial_state: float = 0.0
) -> np.ndarray:
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("Input data must be a 1-dimensional NumPy array.")

    filter_closure = create_filter(alpha, median_N, initial_state)
    y_filtered = np.empty_like(y, dtype=float)

    for i in range(len(y)):
        y_filtered[i] = filter_closure(y[i])

    return y_filtered
