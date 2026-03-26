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
    disturbance_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the closed-loop system iteratively with a time-varying reference array,
    deviation variables, and an anti-windup clamping mechanism.
    """
    time_vector = np.arange(0, t_end, Ts)
    N = len(time_vector)

    # Pre-allocate state arrays. y_0 is expected in Volts.
    y = np.full(N, y_0)
    x_plant = np.full(N, y_0)
    y_hat = np.full(N, y_0)
    u = np.full(N, u_0)
    e = np.zeros(N)

    # Plant ZOH difference equation coefficients
    a_p = np.exp(-Ts / tau_plant)
    b_p = K_plant * (1.0 - a_p)

    # Filter parameters
    filter_median_window = 5
    alpha = 0.88
    # Pre-populate the buffer with the initial steady-state value
    buffer = deque([y_0] * filter_median_window, maxlen=filter_median_window)

    # Controller states
    u_i = 0.0
    e_prev = 0.0

    # Noise
    noise = np.random.normal(0, 0.001, N)

    for n in range(1, N):
        # 1. Plant update
        x_plant[n] = a_p * x_plant[n - 1] + (1.0 - a_p) * y_0 + b_p * (u[n - 1] - u_0)

        # 2. Output formulation with additive disturbance and noise
        y[n] = x_plant[n] + disturbance_array[n] + noise[n]

        # 3. Filter update
        buffer.append(y[n])
        x = np.median(buffer)
        y_hat[n] = alpha * y_hat[n - 1] + (1.0 - alpha) * x

        # 4. Error computation
        e[n] = r_array[n] - y_hat[n]

        # 5. Controller update (Tustin)
        u_p = Kp * e[n]
        u_i_temp = u_i + (Ki * Ts / 2.0) * (e[n] + e_prev)
        v_pi = u_p + u_i_temp  # Deviation control effort (delta_u)

        # 6. Bias addition and Saturation
        v_absolute = v_pi + u_0
        u[n] = np.clip(v_absolute, u_min, u_max)

        # 7. Anti-windup back-calculation
        delta_u_actual = u[n] - u_0
        u_i = delta_u_actual - u_p
        e_prev = e[n]

    return time_vector, y, u, e


def main() -> None:
    # System Parameters
    Ts = 0.1
    K_plant = -0.14
    tau_plant = 35.77
    tau_c = 15.0
    t_end = 1800.0
    sensor_sensitivity = 10.0  # 10 °C per Volt

    # Hardware constraints and operating point
    u_min = 2.0
    u_max = 8.0
    y_0_celsius = 26.0

    # Input that maintains the system at the initial steady-state output in the absence of disturbances
    u_0 = 5.0

    # Controller Design (IMC)
    Kp = tau_plant / (K_plant * tau_c)
    Ki = 1.0 / (K_plant * tau_c)
    print(f"Controller Parameters: Kp = {Kp:.4f}, Ki = {Ki:.4f}")

    # Generate Time Vector
    time_vector = np.arange(0, t_end, Ts)
    N = len(time_vector)

    # Generate Reference Trajectory
    r_array = np.full(N, y_0_celsius)

    ramp_start = t_end * 2 / 5
    ramp_end = t_end * 3 / 5
    ramp_indices = (time_vector >= ramp_start) & (time_vector < ramp_end)
    ramp_length = np.sum(ramp_indices)

    r_array[time_vector >= t_end / 5] = 27.0
    r_array[ramp_indices] = np.linspace(25.0, 26.5, int(ramp_length))
    r_array[time_vector >= t_end * 3 / 5] = 26.5
    r_array[time_vector >= t_end * 4 / 5] = 25.5

    # Generate Disturbance Trajectory
    disturbances = {
        "stepup": (t_end / 5, t_end * 2 / 5, 0.2),  # +0.2 °C step from 1/5 to 2/5
        "stepdown": (
            t_end * 3 / 5,
            t_end * 4 / 5,
            -0.2,
        ),  # -0.2 °C step from 3/5 to 4/5
        "rampup": (t_end / 4, t_end * 3 / 4, 0.2),  # +0.2 °C ramp from 1/4 to 3/4
        "rampdown": (t_end / 3, t_end * 2 / 3, -0.2),  # -0.2 °C ramp from 1/3 to 2/3
        "logup": (
            t_end / 2,
            t_end * 5 / 6,
            0.2,
        ),  # +0.2 °C logarithmic increase from 1/2 to 5/6
        "logdown": (
            t_end / 6,
            t_end / 2,
            -0.2,
        ),  # -0.2 °C logarithmic decrease from 1/6 to 1/2
        "expup": (
            t_end * 6 / 7,
            t_end,
            0.2,
        ),  # +0.2 °C exponential increase from 6/7 to end
        "expdown": (
            t_end / 7,
            t_end * 6 / 7,
            -0.2,
        ),  # -0.2 °C exponential decrease from 1/7 to 6/7
        "sin": (
            t_end * 4 / 9,
            t_end * 7 / 9,
            0.1,
        ),  # 0.1 °C sinusoidal disturbance from 4/9 to 7/9
    }

    # Initialize disturbance array
    disturbance_array = np.zeros(N)
    for name, (start, end, magnitude) in disturbances.items():
        if "step" in name:
            disturbance_array[(time_vector >= start) & (time_vector < end)] += magnitude
        elif "ramp" in name:
            ramp_duration = end - start
            ramp = np.linspace(0, magnitude, int(ramp_duration / Ts))
            disturbance_array[(time_vector >= start) & (time_vector < end)] += ramp
        elif "log" in name:
            log_duration = end - start
            log_time = time_vector[(time_vector >= start) & (time_vector < end)] - start
            log_disturbance = magnitude * np.log1p(log_time / log_duration)
            disturbance_array[
                (time_vector >= start) & (time_vector < end)
            ] += log_disturbance
        elif "exp" in name:
            exp_duration = end - start
            exp_time = time_vector[(time_vector >= start) & (time_vector < end)] - start
            exp_disturbance = magnitude * (1 - np.exp(-exp_time / exp_duration))
            disturbance_array[
                (time_vector >= start) & (time_vector < end)
            ] += exp_disturbance
        elif "sin" in name:
            sin_duration = end - start
            sin_time = time_vector[(time_vector >= start) & (time_vector < end)] - start
            sin_disturbance = magnitude * np.sin(2 * np.pi * sin_time / sin_duration)
            disturbance_array[
                (time_vector >= start) & (time_vector < end)
            ] += sin_disturbance

    # Row disturbance array some indexes back to avoid matching the reference step changes exactly, to better observe the controller's response
    disturbance_array = np.roll(disturbance_array, int(0.235711131719 * ramp_length))

    # Scale reference, initial state, and disturbance to sensor units
    r_array /= sensor_sensitivity
    disturbance_array /= sensor_sensitivity
    y_0_volts = y_0_celsius / sensor_sensitivity

    # Execute Simulation
    time_vector, y, u, e = test(
        Kp=Kp,
        Ki=Ki,
        K_plant=K_plant,
        tau_plant=tau_plant,
        Ts=Ts,
        t_end=t_end,
        u_min=u_min,
        u_max=u_max,
        u_0=u_0,
        y_0=y_0_volts,
        r_array=r_array,
        disturbance_array=disturbance_array,
    )

    # Plot Results
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Convert readings back to physical units for plotting
    y *= sensor_sensitivity
    r_array *= sensor_sensitivity
    disturbance_array *= sensor_sensitivity
    e *= sensor_sensitivity

    ax1.plot(time_vector, y, label="Saída do Sistema", linewidth=2)
    ax1.plot(time_vector, r_array, label="Referência")
    ax1.set_title("Resposta em Malha Fechada da Temperatura")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time_vector, u, label="Sinal de Controle", linewidth=2)
    ax2.set_title("Saída do Controlador")
    ax2.set_ylabel("Tensão (V)")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(time_vector, disturbance_array, label="Distúrbio", linewidth=2)
    ax3.set_title("Distúrbio Aplicado")
    ax3.set_xlabel("Tempo (s)")
    ax3.set_ylabel("Temperatura (°C)")
    ax3.grid(True)
    ax3.legend()

    ax4.plot(time_vector, e, label="Erro de Controle", linewidth=2)
    ax4.set_title("Erro de Controle ao Longo do Tempo")
    ax4.set_xlabel("Tempo (s)")
    ax4.set_ylabel("Erro (°C)")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("degraus.png")
    print("Simulation complete. Profile plotted to 'degraus.png'.")


if __name__ == "__main__":
    main()
