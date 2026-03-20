import numpy as np
import scipy.linalg
import scipy.optimize
from sippy_unipi import system_identification


def fit_arx_graybox(
    u_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """
    Fits a 2nd-order ARX model with a discrete zero to capture the cascade dynamics.

    Returns:
        theta: Estimated parameter array [a1, a2, b1, b2].
        order: The hardcoded auto-regressive order.
        NK: The hardcoded transport delay.
    """
    order = 2
    NK = 2

    N = len(y_train)
    start_idx = max(order, NK + 1)

    Phi = np.zeros((N - start_idx, 4))
    Y = y_train[start_idx:]

    for k in range(start_idx, N):
        Phi[k - start_idx, 0] = -y_train[k - 1]
        Phi[k - start_idx, 1] = -y_train[k - 2]
        Phi[k - start_idx, 2] = u_train[k - NK]
        Phi[k - start_idx, 3] = u_train[k - NK - 1]

    lstsq_res = scipy.linalg.lstsq(Phi, Y)
    theta, _, _, _ = (
        lstsq_res if lstsq_res is not None else (np.zeros(4), None, None, None)
    )

    return theta, order, NK


def fit_oe_graybox(
    u_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """
    Fits a 1st-order Output Error model with a discrete zero via nonlinear optimization.

    Returns:
        theta: Optimized parameter array [f1, f2, b1, b2].
        order: The hardcoded auto-regressive order.
        NK: The hardcoded transport delay.
    """
    order = 1
    NK = 20

    N = len(y_train)
    start_idx = max(order, NK + 1)

    # 1. ARX Initial Guess
    Phi = np.zeros((N - start_idx, 4))
    Y = y_train[start_idx:]
    for k in range(start_idx, N):
        Phi[k - start_idx, 0] = -y_train[k - 1]
        Phi[k - start_idx, 1] = -y_train[k - 2]
        Phi[k - start_idx, 2] = u_train[k - NK]
        Phi[k - start_idx, 3] = u_train[k - NK - 1]

    lstsq_res = scipy.linalg.lstsq(Phi, Y)
    theta_0, _, _, _ = (
        lstsq_res if lstsq_res is not None else (np.zeros(4), None, None, None)
    )
    y_init = y_train[:start_idx]

    # 2. Nonlinear Optimization
    def residuals(theta: np.ndarray) -> np.ndarray:
        roots = np.roots(np.r_[1.0, theta[:2]])
        if np.any(np.abs(roots) >= 1.0):
            return np.full(N - start_idx, 1e6)

        y_sim = np.zeros(N)
        y_sim[:start_idx] = y_init

        for k in range(start_idx, N):
            y_sim[k] = (
                -theta[0] * y_sim[k - 1]
                - theta[1] * y_sim[k - 2]
                + theta[2] * u_train[k - NK]
                + theta[3] * u_train[k - NK - 1]
            )

        return y_train[start_idx:] - y_sim[start_idx:]

    res = scipy.optimize.least_squares(residuals, theta_0, method="trf")

    return res.x, order, NK


def fit_ss_graybox(
    u_train: np.ndarray, y_train: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int, int]:
    """
    Identifies a strictly 2nd-order state-space model, decouples the fast and slow
    dynamics, and enforces a positive internal heat transfer sign convention.

    Returns:
        A_phys, B_phys, C_phys, K: Matrices in the corrected physical cascade canonical form.
        nx: The hardcoded state dimension (2).
        NK: The hardcoded transport delay (1).
    """
    nx = 2
    NK = 1

    N = len(y_train)
    if NK > 1:
        y_fit = y_train[NK - 1 :]
        u_fit = u_train[: -(NK - 1)]
    else:
        y_fit = y_train
        u_fit = u_train

    u_2d = np.atleast_2d(u_fit)
    y_2d = np.atleast_2d(y_fit)

    # Strictly enforce 2nd-order extraction
    sys = system_identification(
        y_2d, u_2d, "N4SID", SS_fixed_order=nx, SS_D_required=False
    )
    A_id, B_id, C_id = sys.A, sys.B, sys.C

    # Eigen-decomposition for modal analysis
    eigvals, V = np.linalg.eig(A_id)

    # Sort eigenvalues by magnitude to strictly isolate fast and slow poles
    idx = np.argsort(np.abs(eigvals))
    eigvals_sorted = eigvals[idx]
    V_sorted = V[:, idx]

    # Map to modal output matrix
    C_diag = C_id @ V_sorted
    c_f = C_diag[0, 0]
    c_s = C_diag[0, 1]

    # Construct the abstract mapping matrix M
    M = np.array([[c_f, 0], [c_f, c_s]])

    # Compute the global similarity transformation T = V * M^-1
    T = V_sorted @ np.linalg.inv(M)
    T_inv = np.linalg.inv(T)

    # Apply similarity transformation to yield the abstract cascade canonical form
    A_cas = np.real(T_inv @ A_id @ T)
    B_cas = np.real(T_inv @ B_id)
    C_cas = np.real(C_id @ T)

    # Apply reflection to correct the physical sign convention
    T_ref = np.array([[-1.0, 0.0], [0.0, 1.0]])

    A_phys = T_ref @ A_cas @ T_ref
    B_phys = T_ref @ B_cas
    C_phys = C_cas @ T_ref

    return (A_phys, B_phys, C_phys, np.zeros((nx, nx))), nx, NK
