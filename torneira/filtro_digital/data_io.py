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


def plot_filtered_data(
    k: np.ndarray, y: np.ndarray, y_filtered: np.ndarray, file_path: str
) -> None:
    """
    Plot the raw and filtered output signals.

    Parameters:
    k (np.ndarray): The time steps of the experiment.
    y (np.ndarray): The raw output signal values.
    y_filtered (np.ndarray): The filtered output signal values.
    file_path (str): The path where the plot will be saved.

    This function creates a single plot comparing the raw and filtered output signals against time steps.
    """

    plt.figure()
    plt.plot(k, y, label="Raw Output Voltage (V)", alpha=0.7)
    plt.plot(k, y_filtered, label="Filtered Output Voltage (V)", alpha=0.7)
    plt.title("Raw vs Filtered Output Voltage vs Time Steps")
    plt.xlabel("Time Steps (k)")
    plt.ylabel("Output Voltage (V)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
