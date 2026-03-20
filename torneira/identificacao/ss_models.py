import numpy as np
from sippy_unipi import system_identification


def fit_ss(
    nx: int, nk: int, u_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identifies a discrete-time stochastic state-space model using N4SID.
    Incorporates explicit transport delay nk >= 1.
    """
    N = len(y_train)
    if len(u_train) != N:
        raise ValueError("u_train and y_train must be of equal length.")
    if nk < 1:
        raise ValueError(
            "nk must be >= 1 for a strictly proper state-space model (D=0)."
        )

    # Shift data arrays to absorb delays > 1 without injecting artificial zeros
    if nk > 1:
        y_fit = y_train[nk - 1 :]
        u_fit = u_train[: -(nk - 1)]
    else:
        y_fit = y_train
        u_fit = u_train

    u_2d = np.atleast_2d(u_fit)
    y_2d = np.atleast_2d(y_fit)

    sys = system_identification(
        y_2d, u_2d, "N4SID", SS_orders=[nx], SS_D_required=False
    )

    return sys.A, sys.B, sys.C, sys.K


def get_states_from_output(C: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Estimate an initial state vector from an output measurement."""
    y0_arr = np.atleast_1d(y0).astype(float)

    if y0_arr.ndim > 1 and y0_arr.shape[1] != 1:
        y0_arr = y0_arr.ravel()

    if y0_arr.ndim == 0:
        y0_arr = np.array([y0_arr])

    if C.ndim != 2:
        raise ValueError("C must be a 2D output matrix")

    if y0_arr.shape[0] != C.shape[0]:
        raise ValueError(
            f"y0 must have length {C.shape[0]}, but has length {y0_arr.shape[0]}"
        )

    x0 = np.linalg.pinv(C) @ y0_arr
    return x0.ravel()


def test_ss(
    nx: int,
    nk: int,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    u_test: np.ndarray,
    y_test: np.ndarray,
    x_init: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Evaluates the deterministic state-space model using a free-run simulation.
    Applies the explicit transport delay step mathematically into the state transition.
    Unmeasured past inputs (k < 0) are assumed to equal the initial input u_test[0].
    """
    A, B, C, K = matrices
    N = len(y_test)

    if len(u_test) != N:
        raise ValueError("u_test and y_test must be of equal length.")
    if len(x_init) != nx:
        raise ValueError("x_init must have length equal to the state dimension nx.")

    x = np.zeros((nx, N + 1))
    x[:, 0] = x_init
    y_pred = np.zeros(N)

    B_vec = B.flatten()

    for k in range(N):
        y_pred[k] = float((C @ x[:, k]).ravel()[0])

        # State transition utilizing the correct delayed input
        u_idx = k + 1 - nk

        # Clamp unmeasured past inputs to the first available input
        u_val = u_test[u_idx] if u_idx >= 0 else u_test[0]

        x[:, k + 1] = A @ x[:, k] + B_vec * u_val

    mse = float(np.mean((y_test - y_pred) ** 2))

    # Return state trajectory excluding the initial state at k=0 so state sequence aligns with y_pred timestamps
    return y_pred, mse, x[:, 1:]


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
