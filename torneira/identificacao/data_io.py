import numpy as np
import pandas as pd
import matplotlib

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


def plot_predictions_poly(
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


def plot_predictions_poly_gb(
    k_test: np.ndarray,
    y_test: np.ndarray,
    arx_result: tuple[np.ndarray, float],
    oe_result: tuple[np.ndarray, float],
    file_path: str,
) -> None:
    """
    Plots polynomial model predictions for greybox comparison using ARX and OE models.

    Parameters:
    k_test (np.ndarray): The time steps for the test data.
    y_test (np.ndarray): The true output signal values for the test data.
    arx_result (tuple[np.ndarray, float]): ARX prediction and MSE, shape (y_pred, mse).
    oe_result (tuple[np.ndarray, float]): OE prediction and MSE, shape (y_pred, mse).
    file_path (str): The path where the plot will be saved.

    This function creates one figure with two subplots side by side:
    - left: ARX prediction vs true output
    - right: OE prediction vs true output
    """

    arx_pred, arx_mse = arx_result
    oe_pred, oe_mse = oe_result

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    ax_arx = axes[0]
    ax_arx.plot(k_test, y_test, label="True Output", color="black", linewidth=2)
    ax_arx.plot(
        k_test,
        arx_pred,
        label=f"ARX Prediction (MSE={arx_mse:.6f})",
        alpha=0.8,
    )
    ax_arx.set_title("ARX Model Prediction vs True Output")
    ax_arx.set_xlabel("Time Steps (k)")
    ax_arx.set_ylabel("Output Voltage (V)")
    ax_arx.grid()
    ax_arx.legend()

    ax_oe = axes[1]
    ax_oe.plot(k_test, y_test, label="True Output", color="black", linewidth=2)
    ax_oe.plot(
        k_test,
        oe_pred,
        label=f"OE Prediction (MSE={oe_mse:.6f})",
        alpha=0.8,
    )
    ax_oe.set_title("OE Model Prediction vs True Output")
    ax_oe.set_xlabel("Time Steps (k)")
    ax_oe.grid()
    ax_oe.legend()

    fig.tight_layout()
    fig.savefig(file_path)


def plot_ss(
    k_test: np.ndarray,
    y_test: np.ndarray,
    ss_results: dict[int, tuple[np.ndarray, float]],
    ss_state_results: dict[int, np.ndarray] | None,
    file_path: str,
) -> None:
    """
    Plots the true output signal and state-space model predictions, plus state trajectories.

    Parameters:
    k_test (np.ndarray): The time steps for the test data.
    y_test (np.ndarray): The true output signal values for the test data.
    ss_results (dict[int, tuple[np.ndarray, float]]): A dictionary where keys are state-space model orders and values are tuples of (predicted output, MSE).
    ss_state_results (dict[int, np.ndarray] | None): A dictionary where keys are state-space orders and values are state trajectories shape (nx, N).
    file_path (str): The path where the output prediction plot will be saved.

    This function creates:
    - one figure comparing y_test with predicted y for each SS model order.
    - one figure plotting all passed state sequences across orders together.
    """

    # Single figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1 = axs[0]
    ax1.plot(k_test, y_test, label="True Output", color="black", linewidth=2)
    for order, (y_pred, mse) in ss_results.items():
        ax1.plot(
            k_test,
            y_pred,
            label=f"SS({order}) Prediction (MSE={mse:.6f})",
            alpha=0.8,
        )
    ax1.set_title("State-Space Model Predictions vs True Output")
    ax1.set_ylabel("Output Voltage (V)")
    ax1.grid()
    ax1.legend()

    ax2 = axs[1]
    if ss_state_results:
        for order, x_traj in ss_state_results.items():
            nx = x_traj.shape[0]
            N = x_traj.shape[1]
            k_states = k_test[:N]
            for state_idx in range(nx):
                ax2.plot(
                    k_states,
                    x_traj[state_idx, :],
                    label=f"SS({order}) x[{state_idx}]",
                    alpha=0.8,
                )
        ax2.set_title("State-Space State Trajectories")
        ax2.set_ylabel("State Value")
        ax2.grid()
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "No SS state trajectories provided",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.set_title("State-Space State Trajectories")
        ax2.set_ylabel("State Value")
        ax2.grid(False)

    ax2.set_xlabel("Time Steps (k)")
    fig.tight_layout()
    fig.savefig(file_path)
