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
    plt.plot(k, u)
    plt.title("Tensão Aplicada vs Amostras")
    plt.xlabel("Amostra")
    plt.ylabel("Tensão Aplicada (V)")
    plt.grid()

    # Plot output signal
    plt.subplot(2, 1, 2)
    plt.plot(k, y)
    plt.title("Tensão Medida vs Amostras")
    plt.xlabel("Amostra")
    plt.ylabel("Tensão Medida (V)")
    plt.grid()

    plt.tight_layout()
    plt.savefig(file_path)


def plot_model(
    k_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    file_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plots the true output signal and model predictions.

    Parameters:
    k_test (np.ndarray): The time steps for the test data.
    y_test (np.ndarray): The true output signal values for the test data.
    y_pred (np.ndarray): The predicted output signal values from the model.
    file_path (str): The path where the plot will be saved.
    ylim (tuple[float, float] | None): The y-axis limits for the plot.

    This function creates a plot comparing the true output signal with the predictions from the model.
    """

    plt.figure()
    plt.plot(k_test, y_test, label="Referência")
    plt.plot(k_test, y_pred, label=f"Inferência")

    if ylim is not None:
        plt.ylim(ylim)

    plt.title("Comparação entre Referência e Inferência do Modelo")
    plt.xlabel("Amostra")
    plt.ylabel("Tensão Medida (V)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
