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
            data["k"].to_numpy(dtype=int),
            data["u"].to_numpy(dtype=float),
            data["y"].to_numpy(dtype=float),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_filter(
    k: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    file_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plots the true output signal and model predictions.

    Parameters:
    k (np.ndarray): The time steps for the data.
    y (np.ndarray): The true output signal values.
    y_hat (np.ndarray): The filtered output signal values.
    file_path (str): The path where the plot will be saved.
    ylim (tuple[float, float], optional): Limits for the y-axis. Defaults to None.

    This function creates a plot comparing the true output signal with the predictions from the model.
    """

    plt.figure()
    plt.plot(k, y, label="Sinal Bruto")
    plt.plot(
        k,
        y_hat,
        label="Sinal Filtrado",
    )

    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel("Amostra")
    plt.ylabel("Tensão Medida (V)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)


def load_model_predictions(file_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load model predictions from a CSV file.

    Parameters:
        file_path: Path to the CSV file.

    Returns:
        (k, y_pred) as NumPy arrays, or None on failure.

    Note: The CSV file is expected to have columns 'k' and 'y_pred'.
    """

    try:
        data = pd.read_csv(file_path)
        # Strip whitespace from column headers (e.g., ' k' -> 'k')
        data.columns = data.columns.str.strip()
        return (
            data["k"].to_numpy(dtype=int),
            data["y_pred"].to_numpy(dtype=float),
        )
    except Exception as e:
        print(f"Error loading model predictions: {e}")
        return None


def compare_residuals(
    k: np.ndarray,
    y_raw: np.ndarray,
    y_median: np.ndarray,
    y_hat: np.ndarray,
    y_ref: np.ndarray,
    file_path: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plots the residuals of the raw signal, median-only filter, and complete filter against a reference signal.

    Parameters:
    k (np.ndarray): The time steps for the data.
    y_raw (np.ndarray): The raw output signal values.
    y_median (np.ndarray): The output signal values from the median-only filter.
    y_hat (np.ndarray): The output signal values from the complete filter.
    y_ref (np.ndarray): The reference signal values for comparison.
    file_path (str): The path where the plot will be saved.
    ylim (tuple[float, float], optional): Limits for the y-axis. Defaults to None.

    This function creates a plot comparing the residuals of the raw signal, median-only filter, and complete filter against a reference signal.
    """

    residual_raw = y_raw - y_ref
    residual_median = y_median - y_ref
    residual_hat = y_hat - y_ref

    plt.figure()
    plt.plot(k, residual_raw, label=f"Resíduo Sinal Bruto", alpha=0.33)
    plt.plot(k, residual_median, label=f"Resíduo Filtro Mediana", alpha=0.66)
    plt.plot(k, residual_hat, label=f"Resíduo Filtro Completo", alpha=0.99)

    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel("Amostra")
    plt.ylabel("Resíduo (V)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_path)
