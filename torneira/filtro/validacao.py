import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("validacao.txt")
    df.columns = df.columns.str.strip()
    k, u, y, y_hat = df["k"], df["u"], df["y"], df["y_hat"]
    k, u, y, y_hat = k[550:1250], u[550:1250], y[550:1250], y_hat[550:1250]

    plt.figure()
    plt.plot(k, y, label="Sinal Bruto")
    plt.plot(k, y_hat, label="Sinal Filtrado")
    plt.xlabel("Amostra")
    plt.ylabel("Tensão (V)")
    plt.ylim(2.1, 3.5)
    plt.legend()
    plt.grid()
    plt.savefig("validacao.png")


if __name__ == "__main__":
    main()
