import numpy as np

from data_io import (
    load_experiment_data,
    plot_data,
    plot_predictions_poly,
    plot_ss,
    plot_predictions_poly_gb,
)
from preprocess import prepare_data
from poly_models import fit_arx, test_arx, fit_oe, test_oe
from ss_models import fit_ss, test_ss, get_states_from_output
from greybox_fit import fit_arx_graybox, fit_oe_graybox, fit_ss_graybox

MODELS = [
    "SS",
    "POLY",
    "GREYBOX",
]  # Set to ["SS"] to test only state-space model, ["POLY"] for polynomial models, ["GREYBOX"] for grey-box models, or any combination of these.


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
        ss_matrices = fit_ss(2, delay_firstorder, u_fit_demeaned, y_fit_demeaned)
        nx_est = ss_matrices[0].shape[0]

        # Estimate initial state from the first test output sample.
        x_init = get_states_from_output(ss_matrices[2], y_test_demeaned[0])

        ss_y_pred, ss_mse, ss_states = test_ss(
            nx_est,
            delay_firstorder,
            ss_matrices,
            u_test_demeaned,
            y_test_demeaned,
            x_init,
        )
        ss_y_pred_remeaned = ss_y_pred + y_fit_mean

        plot_ss(
            k_test,
            y_test,
            {nx_est: (ss_y_pred_remeaned, ss_mse)},
            {nx_est: ss_states},
            "predicoes_modelo_ss.png",
        )

        print(f"SS({nx_est}) MSE: {ss_mse:.4f}")

    if "GREYBOX" in MODELS:
        arx_gb_theta, arx_gb_order, arx_gb_nk = fit_arx_graybox(
            u_fit_demeaned, y_fit_demeaned
        )
        arx_gb_y_pred, arx_gb_mse = test_arx(
            arx_gb_order,
            arx_gb_nk,
            arx_gb_theta,
            u_test_demeaned,
            y_test_demeaned,
            y_test_demeaned[: max(arx_gb_order, arx_gb_nk)],
        )
        arx_gb_y_pred_remeaned = arx_gb_y_pred + y_fit_mean
        print(f"Grey-box ARX MSE: {arx_gb_mse:.4f}")

        oe_gb_theta, oe_gb_order, oe_gb_nk = fit_oe_graybox(
            u_fit_demeaned, y_fit_demeaned
        )
        oe_gb_y_pred, oe_gb_mse = test_oe(
            oe_gb_order,
            oe_gb_nk,
            oe_gb_theta,
            u_test_demeaned,
            y_test_demeaned,
            y_test_demeaned[: max(oe_gb_order, oe_gb_nk)],
        )
        oe_gb_y_pred_remeaned = oe_gb_y_pred + y_fit_mean
        print(f"Grey-box OE MSE: {oe_gb_mse:.4f}")

        plot_predictions_poly_gb(
            k_test,
            y_test,
            (arx_gb_y_pred_remeaned, arx_gb_mse),
            (oe_gb_y_pred_remeaned, oe_gb_mse),
            "predicoes_modelos_poli_greybox.png",
        )

        ss_gb_matrices, ss_gb_order, ss_gb_nk = fit_ss_graybox(
            u_fit_demeaned, y_fit_demeaned
        )
        nx_gb_est = ss_gb_matrices[0].shape[0]
        x_gb_init = get_states_from_output(ss_gb_matrices[2], y_test_demeaned[0])
        ss_gb_y_pred, ss_gb_mse, ss_gb_states = test_ss(
            nx_gb_est,
            ss_gb_nk,
            ss_gb_matrices,
            u_test_demeaned,
            y_test_demeaned,
            x_gb_init,
        )
        ss_gb_y_pred_remeaned = ss_gb_y_pred + y_fit_mean
        print(f"Grey-box SS({nx_gb_est}) MSE: {ss_gb_mse:.4f}")

        plot_ss(
            k_test,
            y_test,
            {nx_gb_est: (ss_gb_y_pred_remeaned, ss_gb_mse)},
            {nx_gb_est: ss_gb_states},
            "predicoes_modelo_ss_greybox.png",
        )


if __name__ == "__main__":
    main()
