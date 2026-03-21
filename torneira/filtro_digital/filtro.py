from design import compute_filter_parameters
from test_filter import validate_filter_performance
from data_io import load_experiment_data, plot_filtered_data
import numpy as np


def main():
    L_model = 1.043  # seconds
    Ts_system = 0.100  # seconds

    N, tau, a = compute_filter_parameters(L=L_model, Ts=Ts_system)

    print("Computed Filter Parameters:")
    print(f"Median Window (N): {N}")
    print(f"IIR Time Constant (tau_f): {tau:.4f} s")
    print(f"IIR Coefficient (alpha): {a:.6f}")
    print(f"IIR Coefficient (1 - alpha): {1 - a:.6f}")

    data = load_experiment_data("../identificacao/experimento.txt")
    if data is None:
        print("Failed to load experimental data. Exiting.")
        return

    k, u, y = data

    results = validate_filter_performance(
        u=u, y=y, median_N=N, alpha=a, L=L_model, Ts=Ts_system, delay_fraction=0.1
    )

    plot_filtered_data(k, y, results["y_filtered"], "filtered_output.png")
    print("Filtered output plot saved as 'filtered_output.png'.")

    valid_start = 1620

    plot_filtered_data(
        k[valid_start:],
        y[valid_start:],
        results["y_filtered"][valid_start:],
        "filtered_output_valid.png",
    )
    print("Filtered output plot for valid region saved as 'filtered_output_valid.png'.")


if __name__ == "__main__":
    main()
