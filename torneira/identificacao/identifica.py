import numpy as np

from data_io import load_experiment_data, plot_data, plot_predictions
from poly_models import fit_arx, test_arx, fit_oe, test_oe


def main():
    data = load_experiment_data("experimento.txt")
    if data is not None:
        k, u, y = data
    else:
        print("Failed to load data. Please check the file path and format.")
        return

    plot_data(k, u, y, "experimento.png")

    valid_start = fit_start = test_start = 1620
    fit_end = 3500
    valid_end = test_end = 5300

    k_valid, u_valid, y_valid = (
        k[valid_start:valid_end],
        u[valid_start:valid_end],
        y[valid_start:valid_end],
    )

    plot_data(k_valid, u_valid, y_valid, "experimento_valido.png")

    k_fit, u_fit, y_fit = (
        k[fit_start:fit_end],
        u[fit_start:fit_end],
        y[fit_start:fit_end],
    )

    plot_data(k_fit, u_fit, y_fit, "experimento_ajuste.png")

    k_test, u_test, y_test = (
        k[test_start:test_end],
        u[test_start:test_end],
        y[test_start:test_end],
    )

    plot_data(k_test, u_test, y_test, "experimento_teste.png")

    u_fit_mean = np.mean(u_fit)
    y_fit_mean = np.mean(y_fit)

    u_fit_demeaned = u_fit - u_fit_mean
    y_fit_demeaned = y_fit - y_fit_mean
    u_test_demeaned = u_test - u_fit_mean
    y_test_demeaned = y_test - y_fit_mean

    orders = [1, 2]

    arx_results = {}
    oe_results = {}

    for order in orders:
        arx_theta = fit_arx(order, u_fit_demeaned, y_fit_demeaned)
        arx_y_pred, arx_mse = test_arx(
            order, arx_theta, u_test_demeaned, y_test_demeaned, y_fit_demeaned[:order]
        )
        arx_y_pred_remeaned = arx_y_pred + y_fit_mean
        print(f"ARX({order}): MSE = {arx_mse:.6f}")
        arx_results[order] = (arx_y_pred_remeaned, arx_mse)
        oe_theta = fit_oe(order, u_fit_demeaned, y_fit_demeaned)
        oe_y_pred, oe_mse = test_oe(
            order, oe_theta, u_test_demeaned, y_test_demeaned, y_fit_demeaned[:order]
        )
        oe_y_pred_remeaned = oe_y_pred + y_fit_mean
        print(f"OE({order}): MSE = {oe_mse:.6f}")
        oe_results[order] = (oe_y_pred_remeaned, oe_mse)

    plot_predictions(
        k_test, y_test, arx_results, oe_results, "predicoes_modelos_polinomiais.png"
    )


if __name__ == "__main__":
    main()
