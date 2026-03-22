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


def plot_filtered_data(
    k: np.ndarray,
    y: np.ndarray,
    y_filtered: np.ndarray,
    file_path: str,
    title: str,
    y_range: tuple[float, float] | None = None,
) -> None:
    """
    Plot the raw and filtered output signals.

    Parameters:
    k (np.ndarray): The time steps of the experiment.
    y (np.ndarray): The raw output signal values.
    y_filtered (np.ndarray): The filtered output signal values.
    file_path (str): The path where the plot will be saved.
    title (str): The title for the plot.
    y_range (tuple[float, float] | None): The valid range for the output signal values.

    This function creates a single plot comparing the raw and filtered output signals against time steps.
    """

    plt.figure()
    if y_range is not None:
        plt.ylim(y_range)
    plt.plot(k, y, label="Raw Output Voltage (V)")
    plt.plot(k, y_filtered, label="Filtered Output Voltage (V)")
    plt.title(title)
    plt.xlabel("Time Steps (k)")
    plt.ylabel("Output Voltage (V)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)


def plot_filtered_data_xrange(
    k: np.ndarray,
    y: np.ndarray,
    y_filtered: np.ndarray,
    file_path: str,
    title: str,
    x_range: tuple[int, int],
) -> None:
    """
    Plot the raw and filtered output signals for a specific range of time steps.

    Parameters:
    k (np.ndarray): The time steps of the experiment.
    y (np.ndarray): The raw output signal values.
    y_filtered (np.ndarray): The filtered output signal values.
    file_path (str): The path where the plot will be saved.
    title (str): The title for the plot.
    x_range (tuple[int, int]): The range of time steps to plot.
    """

    plot_filtered_data(
        k[x_range[0] : x_range[1]],
        y[x_range[0] : x_range[1]],
        y_filtered[x_range[0] : x_range[1]],
        file_path,
        title,
    )
