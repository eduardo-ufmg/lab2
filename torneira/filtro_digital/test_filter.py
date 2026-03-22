import collections
from collections.abc import Callable

import numpy as np


def create_filter(alpha: float, initial_state: float = 0.0) -> Callable[[float], float]:
    """
    Constructs a stateful closure for sample-by-sample hybrid filtering.

    Args:
        alpha: The IIR filter feedback coefficient (pole).
        initial_state: The steady-state value prior to t=0.

    Returns:
        A function that accepts a single float sample and returns the filtered output.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("Pole alpha must strictly fall within the interval [0, 1).")

    # State initialization
    x_buffer = collections.deque([initial_state] * 3, maxlen=3)

    # Dictionary used to permit mutation within the closure scope
    iir_state = {"y_prev": initial_state}
    feedforward_coeff = 1.0 - alpha

    def filter(x: float) -> float:
        x_buffer.append(x)

        # Median operation (N=3)
        x_med = sorted(x_buffer)[1]

        # IIR difference equation
        y = alpha * iir_state["y_prev"] + feedforward_coeff * x_med
        iir_state["y_prev"] = y

        return y

    return filter


def test_filter(alpha: float, y: np.ndarray, initial_state: float = 0.0) -> np.ndarray:
    """
    Instantiates the online hybrid filter and applies it sequentially to a data vector.

    Args:
        alpha: Discrete IIR filter coefficient [0, 1).
        y: 1D array of experimental input samples.
        initial_state: Initial condition for the filter states.

    Returns:
        1D array of filtered samples.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("Input data must be a 1-dimensional NumPy array.")

    # Instantiate the stateful closure
    filter = create_filter(alpha, initial_state)

    # Preallocate output array to match input dimensions
    y_filtered = np.empty_like(y, dtype=float)

    # Apply the filter sample-by-sample to simulate online operation
    for i in range(len(y)):
        y_filtered[i] = filter(y[i])

    return y_filtered
