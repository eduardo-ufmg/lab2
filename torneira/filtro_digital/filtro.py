from design import compute_parameters
from test_filter import test_filter
from data_io import load_experiment_data, plot_filtered_data, plot_filtered_data_xrange


def main():
    L = 1.043  # seconds
    Ts = 0.1  # seconds
    median_N = 5  # Odd integer for median filter window size
    y_range = (2.2, 3.2)  # temperature sensor voltage
    delay_fraction = 0.15  # 15% of the plant delay

    # Compute filter parameters
    tau_f, alpha = compute_parameters(L, Ts, delay_fraction)
    print(
        f"Computed Parameters:\nMedian Filter Window Size: {median_N}\nLow-Pass Filter Time Constant (tau_f): {tau_f:.4f} seconds\nIIR Filter Coefficient (alpha): {alpha:.4f}"
    )

    # Load experimental data
    data_file = "experimento.txt"
    data = load_experiment_data(data_file)
    if data is None:
        print("Failed to load experimental data. Exiting.")
        return
    k, u, y = data

    # Apply the hybrid filter to the experimental data
    y_filtered = test_filter(
        alpha, median_N, y, initial_state=y[0]
    )  # Use the first sample as the initial state for better transient response

    # Plot and save the results
    plot_file = "experimento_filtrado.png"
    plot_filtered_data(
        k, y, y_filtered, plot_file, "Raw vs Filtered Output Voltage", y_range=y_range
    )
    print(f"Filtered data plotted and saved to {plot_file}.")

    # Plot and save a steady-state zoomed-in view
    steady_state_range = (
        3000,
        3500,
    )  # Sample indices corresponding to steady-state region
    plot_file_stead_state = "experimento_filtrado_estacionario.png"
    plot_filtered_data_xrange(
        k,
        y,
        y_filtered,
        plot_file_stead_state,
        "Steady-State Filtered Output Voltage",
        steady_state_range,
    )
    print(f"Steady-state filtered data plotted and saved to {plot_file_stead_state}.")

    # Plot and save a response start zoomed-in view
    response_start_range = (
        1750,
        1850,
    )  # Sample indices corresponding to response start region
    plot_file_response_start = "experimento_filtrado_resposta_inicial.png"
    plot_filtered_data_xrange(
        k,
        y,
        y_filtered,
        plot_file_response_start,
        "Response Start Filtered Output Voltage",
        response_start_range,
    )
    print(
        f"Response start filtered data plotted and saved to {plot_file_response_start}."
    )

    # Plot and save a transient zoomed-in view
    transient_range = (2000, 2500)  # Sample indices corresponding to transient region
    plot_file_transient = "experimento_filtrado_transitorio.png"
    plot_filtered_data_xrange(
        k,
        y,
        y_filtered,
        plot_file_transient,
        "Transient Filtered Output Voltage",
        transient_range,
    )
    print(f"Transient filtered data plotted and saved to {plot_file_transient}.")


if __name__ == "__main__":
    main()
