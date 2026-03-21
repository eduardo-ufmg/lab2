import numpy as np
from scipy.signal import lfilter, correlate


def causal_median_filter_1d(y: np.ndarray, N: int = 3) -> np.ndarray:
    """Applies a strictly causal 1D median filter of window size N."""
    padded_y = np.pad(y, (N - 1, 0), mode="edge")
    # Create sliding windows of size N
    shape = (y.size, N)
    strides = (padded_y.strides[0], padded_y.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded_y, shape=shape, strides=strides)
    return np.median(windows, axis=1)


def validate_filter_performance(
    u: np.ndarray,
    y: np.ndarray,
    median_N: int,
    alpha: float,
    L: float,
    Ts: float,
    delay_fraction: float = 0.1,
) -> dict:
    """
    Validates the hybrid filter against experimental data.

    Returns:
        A dictionary containing the filtered signal, the empirical delay
        introduced, and the noise variance reduction ratio.
    """

    # 2. Apply Causal Median Filter
    y_med = causal_median_filter_1d(y, N=median_N)

    # 3. Apply IIR Low-Pass Filter (Unity DC Gain)
    # Difference equation: y_f[k] - alpha*y_f[k-1] = (1 - alpha)*y_med[k]
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    y_filtered = np.asarray(lfilter(b, a, y_med))

    # 4. Compute Metrics
    # Noise reduction via discrete derivative variance
    dy_raw_var = np.var(np.diff(y))
    dy_filt_var = np.var(np.diff(y_filtered))
    noise_attenuation_ratio = dy_filt_var / dy_raw_var if dy_raw_var > 0 else 1.0

    # Empirical delay addition via cross-correlation with input u
    # Subtract means to avoid DC bias in cross-correlation
    u_centered = u - np.mean(u)

    corr_raw = correlate(y - np.mean(y), u_centered, mode="full")
    lag_raw = np.argmax(corr_raw)

    corr_filt = correlate(y_filtered - np.mean(y_filtered), u_centered, mode="full")
    lag_filt = np.argmax(corr_filt)

    added_delay_samples = lag_filt - lag_raw
    added_delay_seconds = added_delay_samples * Ts

    return {
        "y_filtered": y_filtered,
        "metrics": {
            "noise_variance_ratio": noise_attenuation_ratio,
            "added_delay_seconds": added_delay_seconds,
            "theoretical_max_added_delay": L * delay_fraction,
        },
    }
