import numpy as np

from data_io import (
    load_experiment_data,
    plot_data,
    plot_predictions_poly,
    plot_predictions_ss,
)
from preprocess import prepare_data
from poly_models import fit_arx, test_arx, fit_oe, test_oe
from ss_models import fit_ss, test_ss, get_states_from_output

MODELS = [
    "SS",
    "POLY",
]  # Set to ["SS"] to test only state-space model, ["POLY"] for polynomial models, or both for all models


def main():
    data = load_experiment_data("experimento.txt")
    if data is not None:
        k, u, y = data
    else:
        print("Failed to load data. Please check the file path and format.")
        return

    plot_data(k, u, y, "experimento.png")

    delay_inspection_bounds = (1780, 1880)
    k_delay_inspection = k[delay_inspection_bounds[0] : delay_inspection_bounds[1]]
    u_delay_inspection = u[delay_inspection_bounds[0] : delay_inspection_bounds[1]]
    y_delay_inspection = y[delay_inspection_bounds[0] : delay_inspection_bounds[1]]

    plot_data(
        k_delay_inspection,
        u_delay_inspection,
        y_delay_inspection,
        "inspecao_atraso.png",
    )

    (
        k_test,
        y_test,
        u_fit_demeaned,
        y_fit_demeaned,
        u_test_demeaned,
        y_test_demeaned,
        y_fit_mean,
    ) = prepare_data(k, u, y)

    delay_firstorder = 25
    delay_secondorder = 10

    if "POLY" in MODELS:
        orders = [1, 2]

        arx_results = {}
        oe_results = {}

        delays = {
            1: delay_firstorder,
            2: delay_secondorder,
        }

        for order in orders:

            y_init = y_test_demeaned[: max(order, delays[order])]

            arx_theta = fit_arx(order, delays[order], u_fit_demeaned, y_fit_demeaned)
            arx_y_pred, arx_mse = test_arx(
                order,
                delays[order],
                arx_theta,
                u_test_demeaned,
                y_test_demeaned,
                y_init,
            )
            arx_y_pred_remeaned = arx_y_pred + y_fit_mean
            arx_results[order] = (arx_y_pred_remeaned, arx_mse)

            oe_theta = fit_oe(order, delays[order], u_fit_demeaned, y_fit_demeaned)
            oe_y_pred, oe_mse = test_oe(
                order,
                delays[order],
                oe_theta,
                u_test_demeaned,
                y_test_demeaned,
                y_init,
            )
            oe_y_pred_remeaned = oe_y_pred + y_fit_mean
            oe_results[order] = (oe_y_pred_remeaned, oe_mse)

        plot_predictions_poly(
            k_test, y_test, arx_results, oe_results, "predicoes_modelos_polinomiais.png"
        )

        for order, (_, mse) in arx_results.items():
            print(f"ARX({order}) MSE: {mse:.4f}")

        for order, (_, mse) in oe_results.items():
            print(f"OE({order}) MSE: {mse:.4f}")

    if "SS" in MODELS:
        matrices_ss = fit_ss(2, delay_firstorder, u_fit_demeaned, y_fit_demeaned)
        nx_est = matrices_ss[0].shape[0]

        # Estimate initial state from the first test output sample.
        x_init = get_states_from_output(matrices_ss[2], y_test_demeaned[0])

        ss_y_pred, ss_mse = test_ss(
            nx_est,
            delay_firstorder,
            matrices_ss,
            u_test_demeaned,
            y_test_demeaned,
            x_init,
        )
        ss_y_pred_remeaned = ss_y_pred + y_fit_mean

        plot_predictions_ss(
            k_test,
            y_test,
            {nx_est: (ss_y_pred_remeaned, ss_mse)},
            "predicoes_modelo_ss.png",
        )

        print(f"SS({nx_est}) MSE: {ss_mse:.4f}")


if __name__ == "__main__":
    main()
