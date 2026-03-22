import numpy as np


def compute_parameters(
    L: float, Ts: float, delay_fraction: float = 0.1
) -> tuple[int, float, float]:
    """
    Computes the parameters for the hybrid real-time digital filter.

    Args:
        L: Plant transport delay (dead time) in seconds.
        Ts: Sampling period in seconds.
        delay_fraction: The maximum allowed ratio of filter time constant to plant delay.
                        Defaults to 0.1 (10%) to preserve closed-loop phase margin.

    Returns:
        median_N: Window size for the median filter.
        tau_f: Continuous time constant of the low-pass filter (seconds).
        alpha: Discrete IIR filter coefficient for the previous output.
    """
    if L <= 0 or Ts <= 0:
        raise ValueError(
            "Dead time (L) and sampling period (Ts) must be strictly positive."
        )
    if delay_fraction <= 0 or delay_fraction >= 1:
        raise ValueError("Delay fraction must be within the interval (0, 1).")

    # 1. Median Filter Parameter
    median_N = 7

    # 2. IIR Low-Pass Filter Parameters
    tau_f = L * delay_fraction

    # 3. Exact Z-plane pole mapping
    alpha = float(np.exp(-Ts / tau_f))

    return median_N, tau_f, alpha
