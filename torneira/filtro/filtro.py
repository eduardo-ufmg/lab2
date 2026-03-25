import numpy as np

from data_io import (
    load_experiment_data,
    plot_filter,
    load_model_predictions,
    compare_residuals,
)
from digital_filter import DigitalFilter


def main():
    # Load experimental data
    data = load_experiment_data("experimento.txt")
    if data is None:
        print("Failed to load data. Exiting.")
        return

    k, u, y = data

    # Median-only filter
    median_filter = DigitalFilter(median_window_size=5, alpha=0.0)
    y_median = np.array([median_filter.filter(y_i) for y_i in y])

    # Apply complete digital filter to the data
    complete_filter = DigitalFilter(median_window_size=5, alpha=0.88)
    y_hat = np.array([complete_filter.filter(y_i) for y_i in y])

    # Plot the original and filtered signals
    plot_filter(
        k,
        y,
        y_hat,
        file_path="filtro.png",
        ylim=(2.2, 3.2),
    )

    ref = load_model_predictions("predicoes_modelo.csv")

    if ref is None:
        print("Failed to load model predictions. Exiting.")
        return

    k_ref, y_ref = ref
    ref_start, ref_end = k_ref[0], k_ref[-1]
    constraint_indices = (k >= ref_start) & (k <= ref_end)

    # Compare residuals between the filtered signal and the model predictions
    compare_residuals(
        k[constraint_indices],
        y[constraint_indices],
        y_median[constraint_indices],
        y_hat[constraint_indices],
        y_ref,
        file_path="residuos.png",
        ylim=(-0.05, 0.07),
    )


if __name__ == "__main__":
    main()
