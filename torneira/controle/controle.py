import numpy as np
import matplotlib.pyplot as plt

from collections import deque


def test(
    Kp: float,
    Ki: float,
    K_plant: float,
    tau_plant: float,
    Ts: float,
    t_end: float,
    u_min: float,
    u_max: float,
    u_0: float,
    y_0: float,
    r_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the closed-loop system iteratively with a time-varying reference array,
    deviation variables, and an anti-windup clamping mechanism.
    """
    time_vector = np.arange(0, t_end, Ts)
    N = len(time_vector)

    # Pre-allocate state arrays
    y = np.full(N, y_0)
    y_hat = np.full(N, y_0)
    u = np.full(N, u_0)
    e = np.zeros(N)

    # Plant ZOH difference equation coefficients
    a_p = np.exp(-Ts / tau_plant)
    b_p = K_plant * (1.0 - a_p)

    # Filter parameters
    filter_median_window = 5
    alpha = 0.88
    buffer = deque(maxlen=filter_median_window)

    # Controller states
    u_i = 0.0
    e_prev = 0.0

    for n in range(1, N):
        # 1. Plant update (Deviation formulation)
        y[n] = a_p * y[n - 1] + (1.0 - a_p) * y_0 + b_p * (u[n - 1] - u_0)

        # 2. Filter update
        buffer.append(y[n])
        x = np.median(buffer)
        y_hat[n] = alpha * y_hat[n - 1] + (1.0 - alpha) * x

        # 3. Error computation
        e[n] = r_array[n] - y_hat[n]

        # 4. Controller update (Tustin)
        u_p = Kp * e[n]
        u_i_temp = u_i + (Ki * Ts / 2.0) * (e[n] + e_prev)
        v_pi = u_p + u_i_temp  # Deviation control effort (delta_u)

        # 5. Bias addition and Saturation
        v_absolute = v_pi + u_0
        u[n] = np.clip(v_absolute, u_min, u_max)

        # 6. Anti-windup back-calculation
        delta_u_actual = u[n] - u_0
        u_i = delta_u_actual - u_p
        e_prev = e[n]

    return time_vector, y, u


def main() -> None:
    # System Parameters
    Ts = 0.1
    K_plant = -0.14
    tau_plant = 35.77
    tau_c = 15.0
    t_end = 5400.0

    # Hardware constraints and operating point
    u_min = 2.0
    u_max = 8.0
    y_0 = 27.0
    u_0 = 5.0  # Assumed equilibrium control effort

    # Controller Design (IMC)
    Kp = tau_plant / (K_plant * tau_c)
    Ki = 1.0 / (K_plant * tau_c)
    print(f"Controller Parameters: Kp = {Kp:.4f}, Ki = {Ki:.4f}")

    # Generate Reference Trajectory
    time_vector = np.arange(0, t_end, Ts)
    N = len(time_vector)
    r_array = np.full(N, y_0)

    # Steps
    r_array[time_vector >= t_end / 3] = 27.3
    r_array[time_vector >= 2 * t_end / 3] = 26.7

    # Execute Simulation
    time_vector, y, u = test(
        Kp=Kp,
        Ki=Ki,
        K_plant=K_plant,
        tau_plant=tau_plant,
        Ts=Ts,
        t_end=t_end,
        u_min=u_min,
        u_max=u_max,
        u_0=u_0,
        y_0=y_0,
        r_array=r_array,
    )

    # Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time_vector, y, label="Saída do Sistema")
    ax1.plot(time_vector, r_array, label="Referência")
    ax1.set_title("Resposta em Malha Fechada da Temperatura")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time_vector, u, label="Sinal de Controle")
    ax2.axhline(u_max, color="black", linestyle=":", label="Limite Superior")
    ax2.axhline(u_min, color="black", linestyle=":", label="Limite Inferior")
    ax2.axhline(u_0, color="gray", linestyle="-.", label="Equilíbrio")
    ax2.set_title("Saída do Controlador")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Tensão (V)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("degraus.png")
    print("Simulation complete. Profile plotted to 'degraus.png'.")


if __name__ == "__main__":
    main()
