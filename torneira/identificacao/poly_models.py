import numpy as np
import scipy.linalg
import scipy.optimize


def fit_arx(order: int, u_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Fits an ARX model using Ordinary Least Squares.
    """
    N = len(y_train)
    if len(u_train) != N:
        raise ValueError("u_train and y_train must be of equal length.")
    if N <= order:
        raise ValueError("Training length must exceed the auto-regressive order.")

    Phi = np.zeros((N - order, order + 1))
    Y = y_train[order:]

    for k in range(order, N):
        Phi[k - order, :order] = -y_train[k - order : k][::-1]
        Phi[k - order, order] = u_train[k - 1]

    # scipy.linalg.lstsq uses SVD for robust OLS estimation
    lstsq_result = scipy.linalg.lstsq(Phi, Y)
    theta, _, _, _ = (
        lstsq_result if lstsq_result is not None else (None, None, None, None)
    )

    return theta if theta is not None else np.zeros(order + 1)


def test_arx(
    order: int,
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
    if len(y_init) != order:
        raise ValueError("y_init length must strictly equal 'order'.")

    y_pred = np.zeros(N)
    y_pred[:order] = y_init

    a_params = theta[:order]
    b_1 = theta[order]

    for k in range(order, N):
        y_pred[k] = -np.dot(a_params, y_pred[k - order : k][::-1]) + b_1 * u_test[k - 1]

    mse = float(np.mean((y_test[order:] - y_pred[order:]) ** 2))

    return y_pred, mse


def fit_oe(order: int, u_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Fits an Output Error (OE) model using nonlinear least squares optimization.
    Initial parameters are estimated via an ARX Ordinary Least Squares structure.
    """
    N = len(y_train)

    # 1. ARX OLS for initial guess theta_0
    Phi = np.zeros((N - order, order + 1))
    Y = y_train[order:]
    for k in range(order, N):
        Phi[k - order, :order] = -y_train[k - order : k][::-1]
        Phi[k - order, order] = u_train[k - 1]

    lstq_result = scipy.linalg.lstsq(Phi, Y)
    theta_0, _, _, _ = (
        lstq_result
        if lstq_result is not None
        else (np.zeros(order + 1), None, None, None)
    )

    # 2. Nonlinear Output Error Optimization
    y_init = y_train[:order]

    def residuals(theta: np.ndarray) -> np.ndarray:
        f_params = theta[:order]
        b_1 = theta[order]

        # Stability constraint: Roots of F(z) must be inside the unit circle
        roots = np.roots(np.r_[1.0, f_params])
        if np.any(np.abs(roots) >= 1.0):
            # Return high constant penalty to prevent NaN overflow; corrupts local Jacobian.
            return np.full(N - order, 1e6)

        y_sim = np.zeros(N)
        y_sim[:order] = y_init

        for k in range(order, N):
            y_sim[k] = (
                -np.dot(f_params, y_sim[k - order : k][::-1]) + b_1 * u_train[k - 1]
            )

        return y_train[order:] - y_sim[order:]

    # Trust Region Reflective optimization
    res = scipy.optimize.least_squares(residuals, theta_0, method="trf")

    return res.x


def test_oe(
    order: int,
    theta: np.ndarray,
    u_test: np.ndarray,
    y_test: np.ndarray,
    y_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Evaluates an OE model using a strictly free-run simulation.
    """
    N = len(y_test)
    y_pred = np.zeros(N)
    y_pred[:order] = y_init

    f_params = theta[:order]
    b_1 = theta[order]

    for k in range(order, N):
        y_pred[k] = -np.dot(f_params, y_pred[k - order : k][::-1]) + b_1 * u_test[k - 1]

    mse = float(np.mean((y_test[order:] - y_pred[order:]) ** 2))

    return y_pred, mse
