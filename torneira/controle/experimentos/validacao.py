import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_NAME = "pi"  # Change to "imc" for the IMC experiment


def main():
    df = pd.read_csv(f"experimento_{EXPERIMENT_NAME}.txt")
    df.columns = df.columns.str.strip()
    ref, y_hat, u = df["ref"], df["y_hat"], df["u"]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ref, label="Referência")
    plt.plot(y_hat, label="Resposta do Sistema")
    plt.xlabel("Amostra")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(u, label="Sinal de Controle")
    plt.xlabel("Amostra")
    plt.ylabel("Esforço de Controle (V)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"experimento_{EXPERIMENT_NAME}.png")

    fit_start = 1300
    fit_end = 1800
    _, y_hat_fit, u_fit = ref[fit_start:], y_hat[fit_start:], u[fit_start:]

    y_hat_fit_mean = float(np.mean(y_hat_fit))
    u_fit_mean = float(np.mean(u_fit))

    y_hat_fit = y_hat_fit - y_hat_fit_mean
    u_fit = u_fit - u_fit_mean

    plt.figure()
    plt.plot(y_hat_fit, color="blue", label="y")
    plt.plot(u_fit, color="red", label="u")

    ku_peak0 = 1385
    ky_valley0 = 1462
    ky_valley2 = 1762

    vu_peak = 1.5
    vu_valley = -1.3
    vy_peak = 1.0
    vy_valley = -0.9

    plt.vlines(x=ku_peak0, ymin=-1.4, ymax=1.6, colors="green", label="pico_u")
    plt.vlines(
        x=[ky_valley0, ky_valley2], ymin=-1.4, ymax=1.6, colors="orange", label="vale_y"
    )
    plt.hlines(
        y=[vu_peak, vu_valley],
        xmin=fit_start,
        xmax=fit_end,
        colors="purple",
        label="|u|",
    )
    plt.hlines(
        y=[vy_peak, vy_valley],
        xmin=fit_start,
        xmax=fit_end,
        colors="magenta",
        label="|y|",
    )
    plt.legend(loc=(1.01, 0.5))

    Ku = (vu_peak - vu_valley) / 2
    Ky = (vy_peak - vy_valley) / 2
    M = Ky / Ku

    Pu = (ky_valley2 - ky_valley0) / 2 * 0.1  # seconds

    plt.text(1830, -0.5, f"M = {M:.3f}\nPu = {Pu:.3f} s")

    plt.minorticks_on()
    plt.grid(which="both")

    plt.xlabel("Amostra")
    plt.ylabel("Sinal")

    plt.tight_layout()
    plt.savefig(f"experimento_{EXPERIMENT_NAME}_fit.png")


if __name__ == "__main__":
    main()
