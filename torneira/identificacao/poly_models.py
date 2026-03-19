import numpy as np
import scipy.linalg
import scipy.optimize


def fit_arx(
    order: int, nk: int, u_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    """
    Fits an ARX model using Ordinary Least Squares with an explicit transport delay.
    """
    N = len(y_train)
    if len(u_train) != N:
        raise ValueError("u_train and y_train must be of equal length.")

    start_idx = max(order, nk)
    if N <= start_idx:
        raise ValueError("Training length must exceed max(order, nk).")

    Phi = np.zeros((N - start_idx, order + 1))
    Y = y_train[start_idx:]

    for k in range(start_idx, N):
        Phi[k - start_idx, :order] = -y_train[k - order : k][::-1]
        Phi[k - start_idx, order] = u_train[k - nk]

    # scipy.linalg.lstsq uses SVD for robust OLS estimation
    lstsq_result = scipy.linalg.lstsq(Phi, Y)
    theta, _, _, _ = (
        lstsq_result if lstsq_result is not None else (None, None, None, None)
    )

    return theta if theta is not None else np.zeros(order + 1)


def test_arx(
    order: int,
    nk: int,
    theta: np.ndarray,
    u_test: np.ndarray,
    y_test: np.ndarray,
    y_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Tests an ARX model using a strictly free-run simulation.
    """
    N = len(y_test)
    if len(u_test) != N:
        raise ValueError("u_test and y_test must be of equal length.")

    start_idx = max(order, nk)
    if len(y_init) != start_idx:
        raise ValueError(
            f"y_init length must strictly equal max(order, nk)={start_idx}."
        )

    y_pred = np.zeros(N)
    y_pred[:start_idx] = y_init

    a_params = theta[:order]
    b_1 = theta[order]

    for k in range(start_idx, N):
        y_pred[k] = (
            -np.dot(a_params, y_pred[k - order : k][::-1]) + b_1 * u_test[k - nk]
        )

    mse = float(np.mean((y_test[start_idx:] - y_pred[start_idx:]) ** 2))

    return y_pred, mse


def fit_oe(order: int, nk: int, u_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Fits an Output Error (OE) model using nonlinear least squares optimization.
    """
    N = len(y_train)
    start_idx = max(order, nk)

    # 1. ARX OLS for initial guess theta_0
    Phi = np.zeros((N - start_idx, order + 1))
    Y = y_train[start_idx:]
    for k in range(start_idx, N):
        Phi[k - start_idx, :order] = -y_train[k - order : k][::-1]
        Phi[k - start_idx, order] = u_train[k - nk]

    lstq_result = scipy.linalg.lstsq(Phi, Y)
    theta_0, _, _, _ = (
        lstq_result
        if lstq_result is not None
        else (np.zeros(order + 1), None, None, None)
    )

    # 2. Nonlinear Output Error Optimization
    y_init = y_train[:start_idx]

    def residuals(theta: np.ndarray) -> np.ndarray:
        f_params = theta[:order]
        b_1 = theta[order]

        # Stability constraint: Roots of F(z) must be inside the unit circle
        roots = np.roots(np.r_[1.0, f_params])
        if np.any(np.abs(roots) >= 1.0):
            return np.full(N - start_idx, 1e6)

        y_sim = np.zeros(N)
        y_sim[:start_idx] = y_init

        for k in range(start_idx, N):
            y_sim[k] = (
                -np.dot(f_params, y_sim[k - order : k][::-1]) + b_1 * u_train[k - nk]
            )

        return y_train[start_idx:] - y_sim[start_idx:]

    res = scipy.optimize.least_squares(residuals, theta_0, method="trf")

    return res.x


def test_oe(
    order: int,
    nk: int,
    theta: np.ndarray,
    u_test: np.ndarray,
    y_test: np.ndarray,
    y_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Evaluates an OE model using a strictly free-run simulation.
    """
    N = len(y_test)
    start_idx = max(order, nk)

    y_pred = np.zeros(N)
    y_pred[:start_idx] = y_init

    f_params = theta[:order]
    b_1 = theta[order]

    for k in range(start_idx, N):
        y_pred[k] = (
            -np.dot(f_params, y_pred[k - order : k][::-1]) + b_1 * u_test[k - nk]
        )

    mse = float(np.mean((y_test[start_idx:] - y_pred[start_idx:]) ** 2))

    return y_pred, mse
