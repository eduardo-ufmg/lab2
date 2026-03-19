import numpy as np
from sippy_unipi import system_identification


def fit_ss(
    nx: int, u_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identifies a discrete-time stochastic state-space model using N4SID.

    The feedthrough matrix D is constrained to zero.

    Returns:
        Tuple containing matrices (A, B, C, K).
    """
    N = len(y_train)
    if len(u_train) != N:
        raise ValueError("u_train and y_train must be of equal length.")

    # sippy expects 2D arrays: (n_variables, n_samples)
    u_2d = np.atleast_2d(u_train)
    y_2d = np.atleast_2d(y_train)

    # SS_D_required=False enforces strictly proper system (D=0)
    sys = system_identification(
        y_2d, u_2d, "N4SID", SS_orders=[nx], SS_D_required=False
    )

    return sys.A, sys.B, sys.C, sys.K


def get_states_from_output(C: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Estimate an initial state vector from an output measurement.

    This is a simple state reconstruction step that computes an initial
    state `x0` satisfying:

        y0 ≈ C @ x0

    The solution is computed via the Moore-Penrose pseudo-inverse of `C`.

    Args:
        C: Output matrix of shape (l, n) (l = number of outputs, n = number of states).
        y0: Initial output sample(s). Should be shape (l,) or (l, 1) for multiple outputs
            or scalar for a single-output system.

    Returns:
        x0: Estimated initial state vector of shape (n,).
    """
    y0_arr = np.atleast_1d(y0).astype(float)

    # Ensure we have a column vector with the same number of output dims as C
    if y0_arr.ndim > 1 and y0_arr.shape[1] != 1:
        y0_arr = y0_arr.ravel()

    # If y0 is scalar (single output), make it a 1-element vector
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
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    u_test: np.ndarray,
    y_test: np.ndarray,
    x_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Evaluates the deterministic state-space model (A, B, C) using a free-run simulation.
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

    # Flatten B to 1D for direct vector-scalar multiplication
    B_vec = B.flatten()

    for k in range(N):
        # C is shape (1, nx), x[:, k] is (nx,) -> scalar output
        y_pred[k] = float((C @ x[:, k]).ravel()[0])

        # State update
        x[:, k + 1] = A @ x[:, k] + B_vec * u_test[k]

    mse = float(np.mean((y_test - y_pred) ** 2))

    return y_pred, mse
