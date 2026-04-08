import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("pi.txt")
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
    plt.savefig("experimento_pi.png")


if __name__ == "__main__":
    main()
