import numpy as np
import control as ct
import matplotlib.pyplot as plt


def tune(K: float, tau: float, tau_c: float) -> tuple[float, float]:
    """
    Synthesizes PI controller gains using IMC tuning.
    """
    Kp = tau / (K * tau_c)
    Ki = 1.0 / (K * tau_c)
    return Kp, Ki


def test(
    Kp: float,
    Ki: float,
    K_plant: float,
    tau_plant: float,
    Ts: float,
    t_end: float,
    u_min: float,
    u_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the closed-loop system iteratively to accommodate output saturation and anti-windup.
    """
    time_vector = np.arange(0, t_end, Ts)
    N = len(time_vector)

    # State vectors
    y = np.zeros(N)
    y_hat = np.zeros(N)
    u = np.zeros(N)
    e = np.zeros(N)

    # Plant ZOH difference equation coefficients
    # G(s) = K / (tau*s + 1) -> G(z) = b_p / (z - a_p)
    a_p = np.exp(-Ts / tau_plant)
    b_p = K_plant * (1.0 - a_p)

    # Filter parameters (M-point median filter + exponential smoothing)
    ALPHA = 0.88
    M = 5
    filter_hist = np.zeros(M)
    y_hat_prev = 0.0

    # Controller states
    u_i = 0.0
    e_prev = 0.0

    # Reference signal
    half_n = N // 2
    r = np.concat(np.full(half_n, 1), np.full(N - half_n, -1))

    for n in range(1, N):
        # 1. Plant update
        y[n] = a_p * y[n - 1] + b_p * u[n - 1]

        # 2. Filter update
        filter_hist = np.roll(filter_hist, -1)
        filter_hist[-1] = y[n]
        x = np.median(filter_hist)
        y_hat[n] = ALPHA * y_hat_prev + (1.0 - ALPHA) * x
        y_hat_prev = y_hat[n]

        # 3. Error computation
        e[n] = r[n] - y_hat[n]

        # 4. Controller update (Tustin with Anti-Windup)
        u_p = Kp * e[n]
        u_i_temp = u_i + (Ki * Ts / 2.0) * (e[n] + e_prev)
        v = u_p + u_i_temp

        # Saturation
        u[n] = np.clip(v, u_min, u_max)

        # Anti-windup back-calculation
        u_i = u[n] - u_p
        e_prev = e[n]

    return time_vector, y, u


def plot_response(
    time_vector: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    filename: str = "step_response.png",
) -> None:
    """
    Generates a two-panel plot for system output and control effort.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Output Plot
    ax1.plot(time_vector, y, label="Output y(t)", color="blue")
    ax1.axhline(-0.5, color="red", linestyle="--", label="Reference r(t)")
    ax1.set_title("Closed-Loop Output Response")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()

    # Control Effort Plot
    ax2.plot(time_vector, u, label="Control Effort u(t)", color="green")
    ax2.axhline(8.0, color="black", linestyle=":", label="Limit (8V)")
    ax2.axhline(2.0, color="black", linestyle=":", label="Limit (2V)")
    ax2.set_title("Controller Output")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (V)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main() -> None:
    Ts = 0.1
    K_plant = -0.14
    tau_plant = 35.77
    tau_c = 10.0
    t_end = 100.0
    u_min = 2.0
    u_max = 8.0

    # 1. Design
    Kp, Ki = tune(K_plant, tau_plant, tau_c)
    print(f"Continuous PI Gains: Kp = {Kp:.4f}, Ki = {Ki:.4f}")

    # 2. Test (Non-linear simulation)
    time_vector, y, u = test(Kp, Ki, K_plant, tau_plant, Ts, t_end, u_min, u_max)

    # 3. Plot
    plot_response(time_vector, y, u)
    print("Simulation complete. Plot saved to 'step_response.png'.")


if __name__ == "__main__":
    main()
