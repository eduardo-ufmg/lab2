import numpy as np
import scipy.signal


def evaluate_residuals(
    u: np.ndarray, e: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes normalized residual autocorrelation and input-residual cross-correlation.
    """
    N = len(e)
    u_centered = u - np.mean(u)
    e_centered = e - np.mean(e)

    var_u = np.sum(u_centered**2)
    var_e = np.sum(e_centered**2)

    if var_u == 0 or var_e == 0:
        raise ValueError("Zero variance encountered in input or residual arrays.")

    R_ee_full = scipy.signal.correlate(e_centered, e_centered, mode="full") / var_e
    R_ue_full = scipy.signal.correlate(e_centered, u_centered, mode="full") / np.sqrt(
        var_u * var_e
    )
    lags_full = scipy.signal.correlation_lags(N, N, mode="full")

    idx_ee = np.where((lags_full > 0) & (lags_full <= max_lag))[0]
    idx_ue = np.where((lags_full >= -max_lag) & (lags_full <= max_lag))[0]

    rho_ee = R_ee_full[idx_ee]
    rho_ue = R_ue_full[idx_ue]
    conf_bound = 1.96 / np.sqrt(N)

    return rho_ee, rho_ue, conf_bound


def rank_models(models_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """
    Evaluates and ranks candidate models based on M_total.
    The evaluation lag bound is computed automatically based on dataset length.
    """
    if not models_data:
        raise ValueError("The models_data dictionary is empty.")

    # Extract N from the first model array
    first_u, _ = next(iter(models_data.values()))
    N = len(first_u)

    # Compute maximum lag strictly bounded by N/4 and capped at 512
    max_lag = min(512, N // 4)
    if max_lag < 1:
        raise ValueError(
            f"Dataset length N={N} is too small to compute valid correlation lags."
        )

    results = []

    for model_name, (u, e) in models_data.items():
        if len(u) != N or len(e) != N:
            raise ValueError(
                f"Model {model_name} array lengths do not match the expected N={N}."
            )

        rho_ee, rho_ue, conf_bound = evaluate_residuals(u, e, max_lag)

        m_ee = float(np.max(np.abs(rho_ee)))
        m_ue = float(np.max(np.abs(rho_ue)))
        m_total = max(m_ee, m_ue) / conf_bound

        results.append(
            {
                "Model": model_name,
                "M_ee": m_ee,
                "M_ue": m_ue,
                "M_total": m_total,
                "Conf_Bound": conf_bound,
                "Valid": m_total <= 1.0,
            }
        )

    results.sort(key=lambda x: x["M_total"])

    print(
        f"Evaluation configuration: N = {N}, max_lag = {max_lag}, 95% Bound = {conf_bound:.6f}\n"
    )
    print(f"{'Model':<10} | {'M_ee':<8} | {'M_ue':<8} | {'M_total':<8} | {'Valid'}")
    print("-" * 55)
    for res in results:
        valid_str = "PASS" if res["Valid"] else "FAIL"
        print(
            f"{res['Model']:<10} | {res['M_ee']:.6f} | {res['M_ue']:.6f} | {res['M_total']:.6f} | {valid_str}"
        )

    return results
