from data_io import load_experiment_data, plot_data, plot_model, save_model_predictions
from fit_test import fit, test

Ts = 0.1


def main():
    data = load_experiment_data("experimento.txt")
    if data is not None:
        k, u, y = data
    else:
        print("Failed to load data. Please check the file path and format.")
        return

    plot_data(k, u, y, "experimento.png")

    valid_start = test_start = 1620
    valid_end = test_end = 5400

    fit_start, fit_end = 1800, 3600

    k_valid, u_valid, y_valid = (
        k[valid_start:valid_end],
        u[valid_start:valid_end],
        y[valid_start:valid_end],
    )
    k_fit, u_fit, y_fit = (
        k[fit_start:fit_end],
        u[fit_start:fit_end],
        y[fit_start:fit_end],
    )
    k_test, u_test, y_test = (
        k[test_start:test_end],
        u[test_start:test_end],
        y[test_start:test_end],
    )

    y_0 = y_valid[0]

    plot_data(k_valid, u_valid, y_valid, "experimento_valido.png")

    K_0 = (y_fit[-1] - y_0) / (u_fit[-1] - u_fit[0])
    tau_0 = len(y_fit) * Ts / 5

    K, tau = fit(K_0, tau_0, Ts, u_test, y_test, y_0)

    print(f"Estimated parameters: K = {K:.4f}, tau = {tau:.4f} s")

    y_pred = test(K, tau, Ts, u_test, y_0=y_0)

    plot_model(k_test, y_test, y_pred, "modelo_vs_referencia.png", ylim=(2.2, 3.2))

    save_model_predictions(k_test, y_pred, "predicoes_modelo.csv")


if __name__ == "__main__":
    main()
