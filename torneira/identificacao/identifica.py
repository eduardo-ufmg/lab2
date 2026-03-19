import numpy as np
import pandas as pd
import matplotlib
import scipy.linalg

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
    file_path: str,
) -> None:
    """
    Plots the true output signal and ARX model predictions.

    Parameters:
    k_test (np.ndarray): The time steps for the test data.
    y_test (np.ndarray): The true output signal values for the test data.
    arx_results (dict[int, tuple[np.ndarray, float]]): A dictionary where keys are ARX orders and values are tuples of (predicted output, MSE).
    file_path (str): The path where the plot will be saved.

    This function creates a plot comparing the true output signal with the predictions from different ARX models in a 1x2 layout.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (order, (y_pred, mse)) in enumerate(arx_results.items()):
        axes[idx].plot(k_test, y_test, label="True Output", color="black", linewidth=2)
        axes[idx].plot(k_test, y_pred, label=f"ARX({order}) Prediction (MSE={mse:.6f})")
        axes[idx].set_title(f"ARX({order}) Model Prediction vs True Output")
        axes[idx].set_xlabel("Time Steps (k)")
        axes[idx].set_ylabel("Output Voltage (V)")
        axes[idx].grid()
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(file_path)


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

    for order in orders:
        theta = fit_arx(order, u_fit_demeaned, y_fit_demeaned)
        y_pred, mse = test_arx(
            order, theta, u_test_demeaned, y_test_demeaned, y_fit_demeaned[:order]
        )
        y_pred_remeaned = y_pred + y_fit_mean
        print(f"ARX({order}): MSE = {mse:.6f}")
        arx_results[order] = (y_pred_remeaned, mse)

    plot_predictions(k_test, y_test, arx_results, "predicoes_arx.png")


if __name__ == "__main__":
    main()
