import numpy as np

from data_io import plot_data


def prepare_data(
    k: np.ndarray, u: np.ndarray, y: np.ndarray
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """
    Prepares the dataset by splitting it into fitting, validation, and testing subsets.
    It also demeans the input and output data based on the fitting subset and returns the mean of the fitting output for later re-adding to predictions. The split is defined as follows:
    - Fitting: k=1620 to 3500
    - Testing: k=1620 to 5300
    """
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

    return (
        k_test,
        y_test,
        u_fit_demeaned,
        y_fit_demeaned,
        u_test_demeaned,
        y_test_demeaned,
        float(y_fit_mean),
    )
