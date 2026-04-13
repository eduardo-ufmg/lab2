import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    fit_end = 1750
    _, y_hat_fit, u_fit = ref[fit_start:], y_hat[fit_start:], u[fit_start:]

    y_hat_fit_mean = float(np.mean(y_hat_fit))
    u_fit_mean = float(np.mean(u_fit))

    y_hat_fit = y_hat_fit - y_hat_fit_mean
    u_fit = u_fit - u_fit_mean

    plt.figure()
    plt.plot(y_hat_fit, label="Resposta do Sistema")

    xpeak = 1382
    xvalley = 1462

    plt.vlines(
        x=xpeak, ymin=y_hat_fit.min(), ymax=y_hat_fit.max(), colors="red", label="xpico"
    )
    plt.vlines(
        x=xvalley,
        ymin=y_hat_fit.min(),
        ymax=y_hat_fit.max(),
        colors="red",
        label="xvale",
    )

    P = (xvalley - xpeak) * 2  # samples
    Ps = P * 0.1  # Ts = 0.1 s
    print(f"Período P: {P} amostras, {Ps:.1f} segundos")

    ypeak = 29.0 - y_hat_fit_mean
    yvalley = 27.125 - y_hat_fit_mean

    plt.hlines(y=ypeak, xmin=fit_start, xmax=fit_end, colors="green", label="ypico")
    plt.hlines(y=yvalley, xmin=fit_start, xmax=fit_end, colors="green", label="yvale")

    Ymag = (ypeak - yvalley) / 2
    print(f"Amplitude Y: {Ymag:.1f} °C")

    plt.plot(u_fit, label="Sinal de Controle")

    xpeak_u = 1381
    xvalley_u = 1461

    plt.vlines(
        x=xpeak_u, ymin=u_fit.min(), ymax=u_fit.max(), colors="orange", label="xpico_u"
    )
    plt.vlines(
        x=xvalley_u,
        ymin=u_fit.min(),
        ymax=u_fit.max(),
        colors="orange",
        label="xvale_u",
    )

    dk = Ps * 3
    dt = dk * 0.1  # Ts = 0.1 s
    print(f"Atraso dk: {dk} amostras, {dt:.1f} segundos")

    phi = -2 * np.pi * dt / Ps
    print(f"Atraso em fase: {phi:.2f} radianos, {np.degrees(phi):.1f} graus")

    ypeak_u = 1.5
    yvalley_u = -1.3

    plt.hlines(
        y=ypeak_u, xmin=fit_start, xmax=fit_end, colors="purple", label="ypico_u"
    )
    plt.hlines(
        y=yvalley_u, xmin=fit_start, xmax=fit_end, colors="purple", label="yvale_u"
    )

    Umag = (ypeak_u - yvalley_u) / 2
    print(f"Amplitude U: {Umag:.1f} V")

    M = Ymag / Umag
    print(f"Ganho de Magnitude: {M:.1f} °C/V")

    plt.xlabel("Amostra")
    plt.ylabel("Magnitude")
    plt.minorticks_on()
    plt.grid(which="both")

    plt.tight_layout()
    plt.savefig(f"experimento_{EXPERIMENT_NAME}_fit.png")


if __name__ == "__main__":
    main()
