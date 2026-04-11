import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_experiment_data(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load experimental data from a CSV file.

    Parameters:
        file_path: Path to the CSV file.

    Returns:
        (ref, y_hat, u) as NumPy arrays, or None on failure.

    Note: The CSV file is expected to have columns 'ref', 'y_hat' and 'u'.
    """

    try:
        data = pd.read_csv(file_path)
        # Strip whitespace from column headers (e.g., ' u' -> 'u')
        data.columns = data.columns.str.strip()
        return (
            data["ref"].to_numpy(dtype=float),
            data["y_hat"].to_numpy(dtype=float),
            data["u"].to_numpy(dtype=float),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_comparison(
    ref: np.ndarray,
    y_exp: np.ndarray,
    y_sim: np.ndarray,
    u_exp: np.ndarray,
    u_sim: np.ndarray,
    name: str,
) -> None:
    """Plot experimental vs simulated data for both output and input.

    Parameters:
        ref: Reference signal.
        y_exp: Experimental output.
        y_sim: Simulated output.
        u_exp: Experimental input.
        u_sim: Simulated input.
        name: Name for the saved plot file.
    """
    plt.figure(figsize=(12, 8))

    # Output comparison
    plt.subplot(2, 1, 1)
    plt.plot(ref, label="Referencia", linestyle="--", color="gray")
    plt.plot(y_exp, label="Saída Medida", color="blue")
    plt.plot(y_sim, label="Saída Simulada", color="orange")
    plt.title("Resposta Medida vs Simulada")
    plt.xlabel("Amostra")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid()

    # Input comparison
    plt.subplot(2, 1, 2)
    plt.plot(u_exp, label="Sinal de Controle Medido", color="blue")
    plt.plot(u_sim, label="Sinal de Controle Simulado", color="orange")
    plt.title("Comparação do Sinal de Controle")
    plt.xlabel("Amostra")
    plt.ylabel("Tensão (V)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(name)
