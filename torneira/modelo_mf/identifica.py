import numpy as np
from scipy.optimize import least_squares
from data_io import load_experiment_data, plot_comparison


def simulate_closed_loop_discrete(theta, ref, y0, u0, na, nb):
    """
    Simulates the closed-loop system using a generic discrete OE model.
    theta = [b_0, b_1, ..., b_{nb}, a_1, a_2, ..., a_{na}]
    """
    # Extract polynomials
    b = theta[: nb + 1]
    a = theta[nb + 1 :]

    # Controller coefficients
    q0, q1 = -1.50625, 1.5

    N = len(ref)
    y_sim = np.zeros(N)
    u_sim = np.zeros(N)
    e = np.zeros(N)

    # Initialize states
    y_sim[0] = y0
    u_sim[0] = u0
    e[0] = ref[0] - y0

    for k in range(1, N):
        # A. Compute Plant Output y(k)
        yk = 0.0

        # Autoregressive terms (Denominator: a_1 to a_na)
        for i in range(1, na + 1):
            idx = k - i
            val = y_sim[idx] if idx >= 0 else y0
            yk -= a[i - 1] * val

        # Moving average terms (Numerator: b_0 to b_nb)
        for j in range(0, nb + 1):
            idx = k - j
            val = u_sim[idx] if idx >= 0 else u0
            yk += b[j] * val

        y_sim[k] = yk

        # B. Compute Error e(k)
        e[k] = ref[k] - y_sim[k]

        # C. Compute Controller Output u(k)
        u_calc = u_sim[k - 1] + q0 * e[k] + q1 * e[k - 1]

        # D. Apply Saturation Bounds [2, 8]
        u_sim[k] = np.clip(u_calc, 2.0, 8.0)

    return y_sim, u_sim


def objective_function(theta, ref, y_hat, y0, u0, na, nb):
    """Calculates the residual array for least_squares."""
    y_sim, _ = simulate_closed_loop_discrete(theta, ref, y0, u0, na, nb)

    # Strictly penalize unstable roots (divergent simulations)
    if not np.all(np.isfinite(y_sim)) or np.any(np.abs(y_sim) > 1e6):
        return np.full(len(y_hat), 1e6)

    return y_hat - y_sim


def main():
    data = load_experiment_data("experimento_pi.txt")
    if data is None:
        return

    ref, y_hat, u = data
    ref, y_hat, u = ref[500:], y_hat[500:], u[500:]  # Discard initial transient

    # Extract physical initial conditions
    y0 = y_hat[0]
    u0 = u[0]

    # --- Structure Definition ---
    # na: Number of denominator coefficients (poles/order).
    # nb: Number of numerator coefficients.
    # Set nb large enough to capture the maximum expected discrete delay.
    # E.g., if max expected delay is 3 seconds at Ts=0.1, max d = 30. Set nb = 30 + expected zeros.

    na = 0
    nb = 100

    # Dynamic parameter initialization: [b_0, ..., b_nb, a_1, ..., a_na]
    # Initializing 'a' coefficients to simulate a stable integrator/low-pass behavior is generally safer
    # than zeros to prevent immediate gradient explosions.
    initial_b = np.zeros(nb + 1)
    initial_b[0] = -0.01  # Small negative gain to match known negative Kp
    initial_a = np.zeros(na)
    if na > 0:
        initial_a[0] = -0.9  # Stable pole approximation

    initial_theta = np.concatenate((initial_b, initial_a))

    # Parameter estimation (Unconstrained, relying on instability penalty)
    result = least_squares(
        objective_function,
        initial_theta,
        args=(ref, y_hat, y0, u0, na, nb),
        method="lm",  # Levenberg-Marquardt is often better for unconstrained nonlinear least squares
        max_nfev=5000,
    )

    if not result.success:
        print(f"Optimization failed: {result.message}")
        return

    estimated_theta = result.x
    est_b = estimated_theta[: nb + 1]
    est_a = estimated_theta[nb + 1 :]

    print(f"--- Identification Results (na={na}, nb={nb}) ---")
    print(f"Denominator (a) coefficients: {est_a}")
    print(f"Numerator (b) coefficients:\n{np.round(est_b, 4)}")

    # Simulate with estimated parameters
    y_sim, u_sim = simulate_closed_loop_discrete(estimated_theta, ref, y0, u0, na, nb)

    # Plot comparison
    plot_comparison(ref, y_hat, y_sim, u, u_sim, f"resposta_discreta_na{na}_nb{nb}.png")


if __name__ == "__main__":
    main()
