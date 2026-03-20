import numpy as np
import scipy.optimize


def fit_tf(
    k: np.ndarray, u: np.ndarray, y: np.ndarray, Ts: float
) -> tuple[float, float, float, int]:
    """
    Fits a First-Order Plus Dead Time (FOPDT) model to a noisy step response.

    Args:
        k: Discrete sample index array.
        u: Input vector (Volts).
        y: Measured output vector.
        Ts: Sampling period in seconds.

    Returns:
        K: Process gain.
        tau: Time constant (seconds).
        L: Dead time (seconds).
        step_idx: Index of the step transition.
    """
    if len(k) != len(u) or len(k) != len(y):
        raise ValueError("Arrays k, u, and y must have identical lengths.")

    # 1. Locate the step transition
    # Use np.diff to find the index where u drops from 8 to 2
    step_indices = np.where(np.diff(u) < -3.0)[0]
    if len(step_indices) == 0:
        raise ValueError("No valid negative step (8V to 2V) found in input u.")

    step_idx = step_indices[0] + 1
    t_step = k[step_idx] * Ts
    delta_u = u[-1] - u[0]  # Expected to be ~ -6.0 V

    # 2. Robust Empirical Estimation for Initial Guesses (theta_0)
    # Average the pre-step region to establish the initial baseline
    y_0 = np.mean(y[:step_idx])

    # Average the final 10% of the dataset to establish the final steady state
    tail_length = max(1, len(y) // 10)
    y_f = np.mean(y[-tail_length:])

    K_0 = (y_f - y_0) / delta_u

    # Estimate dead time L_0: time until signal deviates by 5% of total change
    threshold = y_0 + 0.05 * (y_f - y_0)

    # Safeguard: direction of crossing depends on the sign of K
    if y_f > y_0:
        cross_idx = np.argmax(y[step_idx:] > threshold)
    else:
        cross_idx = np.argmax(y[step_idx:] < threshold)

    L_0 = k[step_idx + cross_idx] * Ts - t_step
    if L_0 <= 0:
        L_0 = (
            k[1] * Ts - k[0] * Ts
        )  # Default to one sample period if noise triggers immediate crossing

    # Estimate tau_0: time from L_0 to reach 63.2% of total change
    tau_threshold = y_0 + 0.632 * (y_f - y_0)
    if y_f > y_0:
        tau_idx = np.argmax(y[step_idx:] > tau_threshold)
    else:
        tau_idx = np.argmax(y[step_idx:] < tau_threshold)

    tau_0 = (k[step_idx + tau_idx] * Ts - t_step) - L_0
    if tau_0 <= 0:
        tau_0 = (k[-1] * Ts - t_step) / 5.0  # Fallback heuristic

    theta_0 = [K_0, tau_0, L_0]

    # 3. Formulate the Analytical Step Response
    def fopdt_response(theta: np.ndarray, t_data: np.ndarray) -> np.ndarray:
        K, tau, L = theta
        y_sim = np.full_like(t_data, y_0, dtype=float)

        # Boolean mask for t > t_step + L
        active_mask = t_data > (t_step + L)

        # Calculate exponential rise/decay only for the active region
        t_active = t_data[active_mask]
        y_sim[active_mask] = y_0 + K * delta_u * (
            1.0 - np.exp(-(t_active - t_step - L) / tau)
        )

        return y_sim

    # 4. Objective Function for Least Squares
    def residuals(theta: np.ndarray) -> np.ndarray:
        return y - fopdt_response(theta, k * Ts)

    # 5. Execute Bounded Nonlinear Optimization
    # Bounds: K < 0, tau > 0, L >= 0
    bounds = ([-np.inf, 1e-6, 0.0], [0.0, np.inf, np.inf])

    res = scipy.optimize.least_squares(residuals, theta_0, bounds=bounds, method="trf")

    K_opt, tau_opt, L_opt = res.x

    return float(K_opt), float(tau_opt), float(L_opt), int(step_idx)


def test_tf(
    K: float, tau: float, L: float, u_test: np.ndarray, y_test: np.ndarray, Ts: float
) -> tuple[np.ndarray, float]:
    """
    Evaluates an FOPDT model on a validation dataset using a Zero-Order Hold (ZOH)
    discrete-time difference equation.

    Args:
        K: Process gain.
        tau: Time constant (seconds).
        L: Dead time (seconds).
        u_test: Input vector (absolute values).
        y_test: Measured output vector (absolute values, for MSE calculation).
        Ts: Sampling period in seconds.

    Returns:
        y_pred: Predicted output vector (absolute values).
        mse: Mean Squared Error.
    """
    N = len(y_test)
    if len(u_test) != N:
        raise ValueError("Arrays u_test and y_test must have equal lengths.")

    # 1. ZOH Discretization Parameters
    a = np.exp(-Ts / tau)
    b = K * (1.0 - a)

    # Convert continuous dead time to discrete sample delay
    nk = int(np.round(L / Ts))

    # 2. Establish Equilibrium Operating Points
    u_0 = u_test[0]
    y_0 = y_test[0]

    # Convert absolute inputs to deviation variables
    u_dev = u_test - u_0
    y_pred_dev = np.zeros(N)

    # 3. Simulate Difference Equation
    for k in range(1, N):
        # Apply transport delay (+1 inherent sample delay from strictly proper ZOH)
        u_idx = k - 1 - nk

        # Clamp unmeasured past inputs to the initial steady-state deviation (0.0)
        u_val = u_dev[u_idx] if u_idx >= 0 else 0.0

        y_pred_dev[k] = a * y_pred_dev[k - 1] + b * u_val

    # 4. Reconstruct Absolute Output and Compute Metric
    y_pred = y_pred_dev + y_0
    mse = float(np.mean((y_test - y_pred) ** 2))

    return y_pred, mse
