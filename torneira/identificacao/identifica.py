import numpy as np
import pandas as pd
import matplotlib
import scipy.linalg
import scipy.optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_experiment_data(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load experimental data from a CSV file.

    Parameters:
        file_path: Path to the CSV file.

    Returns:
        (k, u, y) as NumPy arrays, or None on failure.

    Note: The CSV file is expected to have columns 'k', 'u' and 'y'.
    """

    try:
        data = pd.read_csv(file_path)
        # Strip whitespace from column headers (e.g., ' u' -> 'u')
        data.columns = data.columns.str.strip()
        return (
            data["k"].to_numpy(dtype=float),
            data["u"].to_numpy(dtype=float),
            data["y"].to_numpy(dtype=float),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_data(k: np.ndarray, u: np.ndarray, y: np.ndarray, file_path: str) -> None:
    """
    Plot the experimental data for input and output signals.

    Parameters:
    k (np.ndarray): The time steps of the experiment.
    u (np.ndarray): The input signal values.
    y (np.ndarray): The output signal values.
    file_path (str): The path where the plot will be saved.

    This function creates two subplots: one for the input signal and one for the output signal, both plotted against time steps.
    """

    plt.figure()

    # Plot input signal
    plt.subplot(2, 1, 1)
    plt.plot(k, u, label="Input Voltage (V)")
    plt.title("Input Voltage vs Time Steps")
    plt.xlabel("Time Steps (k)")
    plt.ylabel("Input Voltage (V)")
    plt.grid()
    plt.legend()

    # Plot output signal
    plt.subplot(2, 1, 2)
    plt.plot(k, y, label="Temperature Sensor Voltage (V)")
    plt.title("Temperature Sensor Voltage vs Time Steps")
    plt.xlabel("Time Steps (k)")
    plt.ylabel("Temperature Sensor Voltage (V)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)


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


def plot_predictions(
    k_test: np.ndarray,
    y_test: np.ndarray,
    arx_results: dict[int, tuple[np.ndarray, float]],
    oe_results: dict[int, tuple[np.ndarray, float]],
    file_path: str,
) -> None:
    """
    Plots the true output signal, ARX, and OE model predictions.

    Parameters:
    k_test (np.ndarray): The time steps for the test data.
    y_test (np.ndarray): The true output signal values for the test data.
    arx_results (dict[int, tuple[np.ndarray, float]]): A dictionary where keys are ARX orders and values are tuples of (predicted output, MSE).
    oe_results (dict[int, tuple[np.ndarray, float]]): A dictionary where keys are OE orders and values are tuples of (predicted output, MSE).
    file_path (str): The path where the plot will be saved.

    This function creates a plot comparing the true output signal with the predictions from different models in a 2x2 layout.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot ARX results in first row
    for idx, (order, (y_pred, mse)) in enumerate(arx_results.items()):
        axes[0, idx].plot(
            k_test, y_test, label="True Output", color="black", linewidth=2
        )
        axes[0, idx].plot(
            k_test, y_pred, label=f"ARX({order}) Prediction (MSE={mse:.6f})"
        )
        axes[0, idx].set_title(f"ARX({order}) Model Prediction vs True Output")
        axes[0, idx].set_xlabel("Time Steps (k)")
        axes[0, idx].set_ylabel("Output Voltage (V)")
        axes[0, idx].grid()
        axes[0, idx].legend()

    # Plot OE results in second row
    for idx, (order, (y_pred, mse)) in enumerate(oe_results.items()):
        axes[1, idx].plot(
            k_test, y_test, label="True Output", color="black", linewidth=2
        )
        axes[1, idx].plot(
            k_test, y_pred, label=f"OE({order}) Prediction (MSE={mse:.6f})"
        )
        axes[1, idx].set_title(f"OE({order}) Model Prediction vs True Output")
        axes[1, idx].set_xlabel("Time Steps (k)")
        axes[1, idx].set_ylabel("Output Voltage (V)")
        axes[1, idx].grid()
        axes[1, idx].legend()

    plt.tight_layout()
    plt.savefig(file_path)


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


def main():
    data = load_experiment_data("experimento.txt")
    if data is not None:
        k, u, y = data
    else:
        print("Failed to load data. Please check the file path and format.")
        return

    plot_data(k, u, y, "experimento.png")

    valid_start = fit_start = test_start = 1620
    fit_end = 3500
    valid_end = test_end = 5300

    k_valid, u_valid, y_valid = (
        k[valid_start:valid_end],
        u[valid_start:valid_end],
        y[valid_start:valid_end],
    )

    plot_data(k_valid, u_valid, y_valid, "experimento_valido.png")

    k_fit, u_fit, y_fit = (
        k[fit_start:fit_end],
        u[fit_start:fit_end],
        y[fit_start:fit_end],
    )

    plot_data(k_fit, u_fit, y_fit, "experimento_ajuste.png")

    k_test, u_test, y_test = (
        k[test_start:test_end],
        u[test_start:test_end],
        y[test_start:test_end],
    )

    plot_data(k_test, u_test, y_test, "experimento_teste.png")

    u_fit_mean = np.mean(u_fit)
    y_fit_mean = np.mean(y_fit)

    u_fit_demeaned = u_fit - u_fit_mean
    y_fit_demeaned = y_fit - y_fit_mean
    u_test_demeaned = u_test - u_fit_mean
    y_test_demeaned = y_test - y_fit_mean

    orders = [1, 2]

    arx_results = {}
    oe_results = {}

    for order in orders:
        arx_theta = fit_arx(order, u_fit_demeaned, y_fit_demeaned)
        arx_y_pred, arx_mse = test_arx(
            order, arx_theta, u_test_demeaned, y_test_demeaned, y_fit_demeaned[:order]
        )
        arx_y_pred_remeaned = arx_y_pred + y_fit_mean
        print(f"ARX({order}): MSE = {arx_mse:.6f}")
        arx_results[order] = (arx_y_pred_remeaned, arx_mse)
        oe_theta = fit_oe(order, u_fit_demeaned, y_fit_demeaned)
        oe_y_pred, oe_mse = test_oe(
            order, oe_theta, u_test_demeaned, y_test_demeaned, y_fit_demeaned[:order]
        )
        oe_y_pred_remeaned = oe_y_pred + y_fit_mean
        print(f"OE({order}): MSE = {oe_mse:.6f}")
        oe_results[order] = (oe_y_pred_remeaned, oe_mse)

    plot_predictions(k_test, y_test, arx_results, oe_results, "predicoes.png")


if __name__ == "__main__":
    main()
