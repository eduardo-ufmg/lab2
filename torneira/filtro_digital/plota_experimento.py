import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("experimento_validacao.txt")
    k, u, y, y_est = df["k"], df["u"], df["y"], df["y_est"]

    plt.figure(figsize=(10, 6))
    plt.plot(k, u, label="Input Voltage (V)", color="blue")
    plt.plot(k, y, label="Temperature Sensor Voltage (V)", color="orange")
    plt.plot(k, y_est, label="Filtered Output (V)", color="green")
    plt.xlabel("Sample (k)")
    plt.ylabel("Voltage (V)")
    plt.title("Comparison between Actual and Estimated Output")
    plt.legend()
    plt.grid()
    plt.show()
