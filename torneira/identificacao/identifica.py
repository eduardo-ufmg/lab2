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
from model_selection import rank_models

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

    (
        k_test,
        y_test,
        u_fit_demeaned,
        y_fit_demeaned,
        u_test_demeaned,
        y_test_demeaned,
        y_fit_mean,
    ) = prepare_data(k, u, y)

    model_residuals = {}

    if "POLY" in MODELS:
        orders = [1, 2]

        arx_results = {}
        oe_results = {}

        for order in orders:
            arx_theta = fit_arx(order, u_fit_demeaned, y_fit_demeaned)
            arx_y_pred, arx_mse = test_arx(
                order,
                arx_theta,
                u_test_demeaned,
                y_test_demeaned,
                y_test_demeaned[:order],
            )
            arx_y_pred_remeaned = arx_y_pred + y_fit_mean
            arx_results[order] = (arx_y_pred_remeaned, arx_mse)
            arx_residuals = y_test - arx_y_pred_remeaned

            model_residuals[f"ARX({order})"] = (u_test_demeaned, arx_residuals)

            oe_theta = fit_oe(order, u_fit_demeaned, y_fit_demeaned)
            oe_y_pred, oe_mse = test_oe(
                order,
                oe_theta,
                u_test_demeaned,
                y_test_demeaned,
                y_test_demeaned[:order],
            )
            oe_y_pred_remeaned = oe_y_pred + y_fit_mean
            oe_results[order] = (oe_y_pred_remeaned, oe_mse)
            oe_residuals = y_test - oe_y_pred_remeaned

            model_residuals[f"OE({order})"] = (u_test_demeaned, oe_residuals)

        plot_predictions_poly(
            k_test, y_test, arx_results, oe_results, "predicoes_modelos_polinomiais.png"
        )

    if "SS" in MODELS:
        matrices_ss = fit_ss(2, u_fit_demeaned, y_fit_demeaned)
        nx_est = matrices_ss[0].shape[0]

        # Estimate initial state from the first test output sample.
        x_init = get_states_from_output(matrices_ss[2], y_test_demeaned[0])

        ss_y_pred, ss_mse = test_ss(
            nx_est, matrices_ss, u_test_demeaned, y_test_demeaned, x_init
        )
        ss_y_pred_remeaned = ss_y_pred + y_fit_mean
        ss_residuals = y_test - ss_y_pred_remeaned

        model_residuals[f"SS({nx_est})"] = (u_test_demeaned, ss_residuals)

        plot_predictions_ss(
            k_test,
            y_test,
            {nx_est: (ss_y_pred_remeaned, ss_mse)},
            "predicoes_modelo_ss.png",
        )

    ranking = rank_models(model_residuals)
    print("Model Ranking:")
    for rank, model_info in enumerate(ranking, start=1):
        print(
            f"{rank}. {model_info['Model']} - M_total: {model_info['M_total']:.4f} - Valid: {model_info['Valid']}"
        )


if __name__ == "__main__":
    main()
