import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from data_io import load_experiment_data, plot_filter
from digital_filter import DigitalFilter


def main():
    # Load experimental data
    data = load_experiment_data("experimento.txt")
    if data is None:
        print("Failed to load data. Exiting.")
        return

    k, u, y = data

    # Frequency analysis of the model
    K = -0.14  # Volts
    tau = 35.77  # seconds

    # Define the continuous-time transfer function
    num = [K]
    den = [tau, 1]
    system = signal.TransferFunction(num, den)

    # Bode plot
    w, mag, phase = signal.bode(system)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title("Diagrama de Bode do Modelo Identificado")
    plt.ylabel("Magnitude (dB)")
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.tight_layout()
    plt.savefig("bode_modelo.png")

    # Apply digital filter to the data
    digital_filter = DigitalFilter(median_window_size=5, alpha=0.88)
    y_hat = np.array([digital_filter.filter(y_i) for y_i in y])

    # Plot the original and filtered signals
    plot_filter(
        k[1800:3600],
        y[1800:3600],
        y_hat[1800:3600],
        file_path="filtro.png",
        ylim=(2.2, 3.2),
    )


if __name__ == "__main__":
    main()
