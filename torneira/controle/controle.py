from typing import cast

import control
import matplotlib.pyplot as plt
import numpy as np
from control import TransferFunction
from matplotlib.patches import Circle

from digital_controller import DigitalController
from digital_filter import DigitalFilter
from discrete_system import DiscreteSystem

# -----------------------------------------------------------------------------
# 2. Closed-Loop Integration & Validation
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # System Parameters
    K = -1.4
    tau = 35.77
    Ts = 0.1
    u0 = 8.0
    y0 = 23.0

    # Instantiate Components
    plant = DiscreteSystem(K=K, tau=tau, Ts=Ts, u_op=u0, y_op=y0)
    controller = DigitalController(T_s=Ts, tau_cl=tau / 2)
    digital_filter = DigitalFilter()

    # Initialization / Bumpless Transfer Preparation
    # 1. Pre-fill filter buffers to the steady-state operating point
    for _ in range(digital_filter.median_window_size + 1):
        digital_filter.filter(y0)

    # 2. Pre-fill controller history explicitly
    controller.set_auto(False)
    for _ in range(3):
        controller.update_memory(u=u0, e=0.0)
    controller.set_auto(True)

    # Simulation Setup
    time_steps = 24000
    noise_std = 0.2

    # Reference Trajectory Setup
    step0_mag = 8.0
    step0_time_k = 600
    step1_mag = 4.0
    step1_time_k = 7200

    # Disturbance Setup
    dist0_mag = -2.0
    dist0_start_k = 12000
    dist1_mag = 4.0
    dist1_start_k = 17200

    # Data Containers
    r_arr = np.zeros(time_steps)
    y_meas_arr = np.zeros(time_steps)
    y_hat_arr = np.zeros(time_steps)
    u_arr = np.zeros(time_steps)
    e_arr = np.zeros(time_steps)

    # Initial State Iteration
    y_true = y0
    u_curr = u0

    r = y0  # Initial setpoint

    # Execution Loop
    for k in range(time_steps):
        # Generate Reference Trajectory (Step up at k=200)

        if step0_time_k <= k:
            r = y0 + step0_mag
        if step1_time_k <= k:
            r = y0 + step1_mag

        # 1. Plant physical output (true state)
        if k > 0:
            y_true = plant.sample(u_curr)

        # 2. Add disturbance
        if dist0_start_k <= k:
            y_true += dist0_mag
        if dist1_start_k <= k:
            y_true += dist1_mag

        # 3. Sensor measurement with additive Gaussian noise
        y_meas = y_true + np.random.normal(0, noise_std)

        # 4. Filtering pipeline
        y_hat = digital_filter.filter(y_meas)

        # 5. Error computation
        e = r - y_hat

        # 6. Controller output
        u_curr = controller.compute(e)

        # Log Data
        r_arr[k] = r
        y_meas_arr[k] = y_meas
        y_hat_arr[k] = y_hat
        u_arr[k] = u_curr
        e_arr[k] = e

    # -------------------------------------------------------------------------
    # 3. Plotting Results
    # -------------------------------------------------------------------------

    t = np.arange(time_steps) * Ts

    plt.figure()

    # Output Plot
    plt.subplot(2, 1, 1)
    plt.plot(t, y_hat_arr, color="orange", label="Temperatura Estimada ($\\hat{y}$)")
    plt.plot(t, r_arr, color="red", label="Referência ($r$)")
    plt.title("Saída do Sistema em Malha Fechada")
    plt.ylabel("Temperatura (°C)")
    plt.xlabel("Tempo (s)")
    plt.legend()

    # Control Signal Plot
    plt.subplot(2, 1, 2)
    plt.step(t, u_arr, color="green", label="Esforço de Controle ($u$)")
    plt.title("Saída do Controlador Digital")
    plt.ylabel("Tensão (V)")
    plt.xlabel("Tempo (s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resposta_malha_fechada.png")

    # -------------------------------------------------------------------------
    # 5. Closed-Loop Poles and Zeros
    # -------------------------------------------------------------------------
    G = plant.get_tf()
    F = digital_filter.get_tf()
    C = controller.get_tf()

    print(f"Função de Transferência do Sistema Discretizado:\n{G}\n")
    print(f"Função de Transferência do Filtro Digital:\n{F}\n")
    print(f"Função de Transferência do Controlador Digital:\n{C}\n")

    L = C * G
    closed_loop_tf = cast(TransferFunction, control.feedback(L, F))  # type: ignore can you believe sys2 is hinted as strictly int?

    print(f"Função de Transferência em Malha Fechada:\n{closed_loop_tf}\n")

    poles = closed_loop_tf.poles()
    zeros = closed_loop_tf.zeros()

    print(f"Polos do Sistema em Malha Fechada: {poles}")
    print(f"Zeros do Sistema em Malha Fechada: {zeros}")

    plt.figure()
    plt.scatter(np.real(poles), np.imag(poles), marker="x", color="red", label="Polos")
    plt.scatter(np.real(zeros), np.imag(zeros), marker="o", color="blue", label="Zeros")
    plt.gca().add_patch(
        Circle(
            (0, 0),
            1,
            color="black",
            fill=False,
            linestyle="--",
            label="Limite de Estabilidade (|z|=1)",
        )
    )
    plt.title("Polos e Zeros do Sistema em Malha Fechada")
    plt.xlabel("$\\mathbb{R}$")
    plt.ylabel("$\\mathbb{I}$")
    plt.xlim(np.min(np.real(poles)) - 0.01, np.max(np.real(poles)) + 0.01)
    plt.ylim(np.min(np.imag(poles)) - 0.01, np.max(np.imag(poles)) + 0.01)
    plt.grid()
    plt.legend()
    plt.savefig("polos_zeros_malha_fechada.png")

    # -------------------------------------------------------------------------
    # 6. Simplified Closed-Loop Transfer Function
    # -------------------------------------------------------------------------
    snum = [0.01 / 1.4]
    sden = [1, -0.993]

    S = control.TransferFunction(snum, sden, dt=Ts)

    tstep = t[:2000]  # Simulate for the first 200 seconds

    _, sresponse = control.step_response(S, T=tstep)
    _, tresponse = control.step_response(closed_loop_tf, T=tstep)

    plt.figure()
    plt.plot(
        tstep,
        cast(np.ndarray, sresponse),
        label="Resposta ao Degrau do Sistema Simplificado",
    )
    plt.plot(
        tstep,
        cast(np.ndarray, tresponse),
        label="Resposta ao Degrau do Sistema em Malha Fechada",
    )
    plt.title("Comparação de Respostas ao Degrau")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.savefig("comparacao_resposta_degrau.png")
