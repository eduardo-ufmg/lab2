import numpy as np
import scipy.optimize


def test(K: float, tau: float, Ts: float, u_test: np.ndarray, y_0: float) -> np.ndarray:
    """
    Simulates system using ZOH discretization in deviation variables.
    """
    y = np.empty_like(u_test, dtype=float)
    if y.size == 0:
        return y

    a = np.exp(-Ts / tau)
    b = K * (1.0 - a)

    y[0] = y_0
    u_0 = u_test[0]

    for i in range(1, len(y)):
        y[i] = a * y[i - 1] + (1.0 - a) * y_0 + b * (u_test[i - 1] - u_0)

    return y


def fit(
    K_0: float, tau_0: float, Ts: float, u: np.ndarray, y: np.ndarray, y_0: float
) -> tuple[float, float]:
    """
    Fits a model to a noisy step response.
    """

    def residuals(params: np.ndarray) -> np.ndarray:
        K, tau = params
        y_pred = test(K, tau, Ts, u, y_0=y_0)
        return y_pred - y

    res = scipy.optimize.least_squares(residuals, x0=[K_0, tau_0])

    return float(res.x[0]), float(res.x[1])
