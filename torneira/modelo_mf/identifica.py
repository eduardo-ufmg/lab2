import numpy as np
from scipy.optimize import least_squares
from data_io import load_experiment_data, plot_comparison

Ts = 0.1

Pu_seconds = 16.0
wu = (2 * np.pi) / Pu_seconds

Kinit = -1.4
M = 0.7
phi = -1.88

# Initial algebraic guesses
tauinit = (1 / wu) * np.sqrt((Kinit / M) ** 2 - 1)
Linit = (-phi - np.arctan(tauinit * wu)) / wu


def simulate(theta, r_data, Ts, u0, y0):
    """
    Computes the error between the measured plant output and a closed-loop
    simulation using the current FOPDT parameter vector theta.
    """
    K, tau, L = theta
    N = len(r_data)
    y_sim = np.zeros(N)
    u_sim = np.zeros(N)
    e = np.zeros(N)

    # Initialize at measured steady-state
    y_sim[0] = y0
    u_sim[0] = u0
    e[0] = r_data[0] - y0

    # Discretize FOPDT (Zero-Order Hold equivalent using difference equation)
    a = np.exp(-Ts / tau)
    b = K * (1 - a)
    d = int(max(0, np.round(L / Ts)))

    # Original DigitalControllerPI parameters
    Kp = -1.5
    Ti = 24.0
    Ki = Kp / Ti
    u_min, u_max = 2.0, 8.0

    for k in range(1, N):
        # 1. Process Update (Deviation Variables)
        u_past = u_sim[k - 1 - d] if (k - 1 - d) >= 0 else u0
        y_sim[k] = y0 + a * (y_sim[k - 1] - y0) + b * (u_past - u0)

        # 2. Controller Update (Original Incremental Logic)
        e[k] = r_data[k] - y_sim[k]
        du_p = Kp * (e[k] - e[k - 1])
        du_i = Ki * e[k] * Ts

        v_new = u_sim[k - 1] + du_p + du_i

        # Anti-windup (Conditional Integration)
        sat_high = v_new > u_max
        sat_low = v_new < u_min

        if (sat_high and e[k] > 0) or (sat_low and e[k] < 0):
            v_new = u_sim[k - 1] + du_p

        u_sim[k] = max(u_min, min(u_max, v_new))

    return y_sim, u_sim


def objective_function(theta, y_data, r_data, Ts, u0, y0):
    y_sim, _ = simulate(theta, r_data, Ts, u0, y0)
    return y_sim - y_data  # Residuals for least squares


def main():

    # Optimization Formulation
    theta_init = [Kinit, tauinit, Linit]

    print(
        f"Initial Parameter Guesses:\nK = {theta_init[0]:.4f}\ntau = {theta_init[1]:.4f}\nL = {theta_init[2]:.4f}"
    )

    # Bounds: Process is reverse-acting (K < 0). Time constants strictly positive.
    lower_bounds = [-np.inf, 1e-3, 0.0]
    upper_bounds = [0.0, np.inf, np.inf]

    # Load experimental arrays
    data = load_experiment_data("experimento_pi.txt")
    if data is None:
        raise ValueError("Failed to load data. Please check the file path and format.")

    ref, y_hat, u = data

    # Non-Linear Least Squares
    result = least_squares(
        objective_function,
        x0=theta_init,
        bounds=(lower_bounds, upper_bounds),
        args=(y_hat, ref, Ts, u[0], y_hat[0]),
        loss="soft_l1",  # Soft L1 loss adds robustness against high-frequency harmonic discrepancies
    )

    K_opt, tau_opt, L_opt = result.x
    print(
        f"Optimized Parameters:\nK = {K_opt:.4f}\ntau = {tau_opt:.4f}\nL = {L_opt:.4f}"
    )

    # Simulate with optimized parameters for comparison
    y_sim_opt, u_sim_opt = simulate(result.x, ref, Ts, u[0], y_hat[0])

    plot_comparison(
        ref, y_hat, y_sim_opt, u, u_sim_opt, "resposta_simulada_vs_medida.png"
    )


if __name__ == "__main__":
    main()
