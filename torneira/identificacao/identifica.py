import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


def load_experiment_data(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load experimental data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the experimental data.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray] | None: A tuple containing the loaded data as NumPy arrays, or None if an error occurs.

    Note: The CSV file is expected to have columns 'k', 'u', and 'y'.
    """

    try:
        data = pd.read_csv(file_path)
        # Strip whitespace from column headers (e.g., ' u' -> 'u')
        data.columns = data.columns.str.strip()
        return data["k"].to_numpy(), data["u"].to_numpy(), data["y"].to_numpy()
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


def trim_data(
    k: np.ndarray, u: np.ndarray, y: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trim the experimental data to a specified range of time steps.

    Parameters:
    k (np.ndarray): The time steps of the experiment.
    u (np.ndarray): The input signal values.
    y (np.ndarray): The output signal values.
    start (int): The starting time step for trimming.
    end (int): The ending time step for trimming.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the trimmed data as NumPy arrays.

    Note: This function assumes that the time steps in 'k' are sorted and that 'start' and 'end' are valid indices within the range of 'k'.
    """

    mask = (k >= start) & (k <= end)
    return k[mask], u[mask], y[mask]


def fit_first_order(
    u: np.ndarray,
    y: np.ndarray,
    start: int | None = None,
    end: int | None = None,
    enforce_negative_gain: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a first-order discrete-time model to the data.

    Model:
        y[k+1] = a * y[k] + b * u[k]

    Parameters:
        start: The starting sample index (inclusive) to use for fitting.
        end: The ending sample index (inclusive) to use for fitting.
        enforce_negative_gain: If True, force the input coefficient (b) to be non-positive.

    Returns:
        (params, k_fit, y_fit, y_pred_fit)
        where params = [a, b], and y_pred_fit is the model output on the fitting interval.
        k_fit is an array of sample indices for that interval.
    """

    start_idx = 0 if start is None else start
    end_idx = len(u) - 1 if end is None else end

    if start_idx < 0 or end_idx < start_idx or end_idx >= len(u):
        raise ValueError("Invalid start/end indices for fitting.")

    k_fit = np.arange(start_idx, end_idx + 1)
    u_fit = u[start_idx : end_idx + 1]
    y_fit = y[start_idx : end_idx + 1]

    if len(k_fit) < 2:
        raise ValueError("Need at least 2 data points to fit a first-order model.")

    y_k = y_fit[:-1]
    y_k1 = y_fit[1:]
    u_k = u_fit[:-1]

    Phi = np.column_stack((y_k, u_k))

    if enforce_negative_gain:
        lb = [-np.inf, -np.inf]
        ub = [np.inf, 0.0]
        res = lsq_linear(Phi, y_k1, bounds=(lb, ub), lsmr_tol="auto")
        theta = res.x
    else:
        theta, *_ = np.linalg.lstsq(Phi, y_k1, rcond=None)

    y_pred = np.empty_like(y_fit)
    y_pred[0] = y_fit[0]
    for i in range(1, len(y_fit)):
        y_pred[i] = theta[0] * y_pred[i - 1] + theta[1] * u_fit[i - 1]

    return theta, k_fit, y_fit, y_pred


def test_first_order(
    theta: np.ndarray, u_after: np.ndarray, y_last_train: float
) -> np.ndarray:
    """Test a first-order model by predicting forward from the end of the training interval.

    Parameters:
        theta: Model parameters [a, b].
        u_after: Input values for the prediction interval (u[k] for k = k_train_end..k_test_end-1).
        y_last_train: The last measured output from the training interval.

    Returns:
        The predicted output for the test interval (y[k_train_end+1...]).
    """

    y_pred = np.empty(len(u_after) + 1)
    y_pred[0] = y_last_train
    for i in range(1, len(y_pred)):
        y_pred[i] = theta[0] * y_pred[i - 1] + theta[1] * u_after[i - 1]

    return y_pred[1:]


def fit_second_order(
    u: np.ndarray,
    y: np.ndarray,
    start: int | None = None,
    end: int | None = None,
    enforce_negative_gain: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a second-order discrete-time model to the data.

    Model:
        y[k+2] = a1*y[k+1] + a2*y[k] + b1*u[k+1] + b2*u[k]

    Parameters:
        start: The starting sample index (inclusive) to use for fitting.
        end: The ending sample index (inclusive) to use for fitting.
        enforce_negative_gain: If True, force the input coefficients (b1, b2) to be non-positive.

    Returns:
        (params, k_fit, y_fit, y_pred_fit)
        where params = [a1, a2, b1, b2], and y_pred_fit is the model output on the fitting interval.
        k_fit is an array of sample indices for that interval.
    """

    start_idx = 0 if start is None else start
    end_idx = len(u) - 1 if end is None else end

    if start_idx < 0 or end_idx < start_idx or end_idx >= len(u):
        raise ValueError("Invalid start/end indices for fitting.")

    k_fit = np.arange(start_idx, end_idx + 1)
    u_fit = u[start_idx : end_idx + 1]
    y_fit = y[start_idx : end_idx + 1]

    if len(k_fit) < 3:
        raise ValueError("Need at least 3 data points to fit a second-order model.")

    y_k = y_fit[:-2]
    y_k1 = y_fit[1:-1]
    y_k2 = y_fit[2:]
    u_k = u_fit[:-2]
    u_k1 = u_fit[1:-1]

    Phi = np.column_stack((y_k1, y_k, u_k1, u_k))

    if enforce_negative_gain:
        lb = [-np.inf, -np.inf, -np.inf, -np.inf]
        ub = [np.inf, np.inf, 0.0, 0.0]
        res = lsq_linear(Phi, y_k2, bounds=(lb, ub), lsmr_tol="auto")
        theta = res.x
    else:
        theta, *_ = np.linalg.lstsq(Phi, y_k2, rcond=None)

    y_pred = np.empty_like(y_fit)
    y_pred[0] = y_fit[0]
    y_pred[1] = y_fit[1]

    for i in range(2, len(y_fit)):
        y_pred[i] = (
            theta[0] * y_pred[i - 1]
            + theta[1] * y_pred[i - 2]
            + theta[2] * u_fit[i - 1]
            + theta[3] * u_fit[i - 2]
        )

    return theta, k_fit, y_fit, y_pred


def test_second_order(
    theta: np.ndarray, u_after: np.ndarray, y_last_two: tuple[float, float]
) -> np.ndarray:
    """Test a second-order model by predicting forward from the end of the training interval.

    Parameters:
        theta: Model parameters [a1, a2, b1, b2].
        u_after: Input values for the prediction interval.
            Must include u[k_train_end-1] through u[k_test_end-1].
        y_last_two: Tuple (y[k_train_end-1], y[k_train_end]).

    Returns:
        The predicted output for the test interval (y[k_train_end+1...]).
    """

    y_prev, y_last = y_last_two
    y_pred = np.empty(len(u_after) + 1)
    y_pred[0] = y_prev
    y_pred[1] = y_last

    for i in range(2, len(y_pred)):
        y_pred[i] = (
            theta[0] * y_pred[i - 1]
            + theta[1] * y_pred[i - 2]
            + theta[2] * u_after[i - 1]
            + theta[3] * u_after[i - 2]
        )

    return y_pred[2:]


def plot_model_subplots(
    k: np.ndarray,
    y: np.ndarray,
    y_preds: list[np.ndarray],
    labels: list[str],
    title: str,
    file_path: str,
) -> None:
    """Plot measured output vs. model predictions in stacked subplots.

    Each prediction is plotted as a separate subplot, one below the other.
    """

    n = len(y_preds)
    plt.figure(figsize=(8, 3 * n))

    for i, (y_pred, label) in enumerate(zip(y_preds, labels), start=1):
        ax = plt.subplot(n, 1, i)
        ax.plot(k, y, label="Measured Output")
        ax.plot(k, y_pred, label=label, linestyle="--")
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("Time Steps (k)")
        ax.set_ylabel("Temperature Sensor Voltage (V)")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.savefig(file_path)


def main():
    file_path = "experimento.txt"
    data = load_experiment_data(file_path)

    if data is None:
        print("Failed to load experiment data.")
        return

    _, u, y = data
    k = np.arange(len(y))  # k is assumed to be sample index

    plot_data(k, u, y, "experimento.png")

    train_start = 1700
    train_end = 3500

    # Center the data using the training interval (to improve numeric conditioning).
    u_offset = u[train_start : train_end + 1].mean()
    y_offset = y[train_start : train_end + 1].mean()

    u_centered = u - u_offset
    y_centered = y - y_offset

    # Fit models on the training interval (inclusive)
    theta1, k_fit1, y_fit1, y_pred_fit1 = fit_first_order(
        u_centered, y_centered, start=train_start, end=train_end
    )
    theta2, k_fit2, y_fit2, y_pred_fit2 = fit_second_order(
        u_centered, y_centered, start=train_start, end=train_end
    )

    print("First-order model parameters (a, b):", theta1)
    print("Second-order model parameters (a1, a2, b1, b2):", theta2)

    # Prepare the test set as everything after the training interval
    k_test = np.arange(train_end + 1, len(y))
    y_test = y_centered[train_end + 1 :]

    if len(k_test) > 0:
        # First-order: use u[train_end]..u[-2] to predict y[train_end+1]..y[-1]
        u_after_1st = u_centered[train_end:-1]
        y_pred_test_1 = test_first_order(theta1, u_after_1st, y_centered[train_end])

        # Second-order: use u[train_end-1]..u[-2] to predict y[train_end+1]..y[-1]
        u_after_2nd = u_centered[train_end - 1 : -1]
        y_pred_test_2 = test_second_order(
            theta2,
            u_after_2nd,
            (y_centered[train_end - 1], y_centered[train_end]),
        )
    else:
        y_pred_test_1 = np.array([])
        y_pred_test_2 = np.array([])

    # Re-add offsets for plotting
    y_fit1_plot = y_fit1 + y_offset
    y_pred_fit1_plot = y_pred_fit1 + y_offset
    y_test_plot = y_test + y_offset
    y_pred_test_1_plot = y_pred_test_1 + y_offset
    y_pred_test_2_plot = y_pred_test_2 + y_offset

    plot_data(
        k_fit1,
        u[train_start : train_end + 1],
        y_fit1_plot,
        "experimento_sem_transiente.png",
    )

    # Save a single figure for train fit with the two models as stacked subplots.
    plot_model_subplots(
        k_fit1,
        y_fit1_plot,
        [y_pred_fit1_plot, y_pred_fit2 + y_offset],
        ["1st Order Model (train)", "2nd Order Model (train)"],
        "Training fit",
        "fit_train.png",
    )

    if len(k_test) > 0:
        plot_model_subplots(
            k_test,
            y_test_plot,
            [y_pred_test_1_plot, y_pred_test_2_plot],
            ["1st Order Model (test)", "2nd Order Model (test)"],
            "Test prediction",
            "fit_test.png",
        )


if __name__ == "__main__":
    main()
