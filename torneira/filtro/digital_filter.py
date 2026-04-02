from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DigitalFilter:
    """A simple digital filter for processing time-series data."""

    median_window_size: int = 5
    alpha: float = 0.88

    def __post_init__(self):
        """Initialize any additional attributes if necessary."""
        self.buffer = deque(maxlen=self.median_window_size)

    def median(self, y: float) -> float:
        """Calculate the median of the current buffer."""
        self.buffer.append(y)
        return float(np.median(self.buffer))

    def first_order(self, x: float) -> float:
        """Apply a first-order low-pass filter to the input signal."""
        if not hasattr(self, "last_output"):
            self.last_output = x
        output = self.alpha * self.last_output + (1 - self.alpha) * x
        self.last_output = output
        return output

    def filter(self, y: float) -> float:
        """Apply the digital filter to the input signal.

        Parameters:
            y: The current input signal value.

        Returns:
            The filtered output signal value.
        """
        x = self.median(y)
        y_hat = self.first_order(x)
        return y_hat
