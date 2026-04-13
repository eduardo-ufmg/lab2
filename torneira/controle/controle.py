from typing import cast

import control
import matplotlib.pyplot as plt
import numpy as np
from control import TransferFunction
from digital_controller import DigitalControllerIMC, DigitalControllerPID
from digital_filter import DigitalFilter
from discrete_system import DiscreteSystem
from matplotlib.patches import Circle

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

    # Instantiate Components for the IMC simulation
    plantIMC = DiscreteSystem(K=K, tau=tau, Ts=Ts, u_op=u0, y_op=y0)
    controllerIMC = DigitalControllerIMC(tau_cl=tau / 2, T_s=Ts)
    dfIMC = DigitalFilter()

    # Instantiate Components for the PID simulation
    plantPID = DiscreteSystem(K=K, tau=tau, Ts=Ts, u_op=u0, y_op=y0)
    controllerPID = DigitalControllerPID(Kp=-1.5, Ti=24.0, Td=0.0, N=0, T_s=Ts)
    dfPID = DigitalFilter()

    # Initialization / Bumpless Transfer Preparation
    # 1. Pre-fill filter buffers to the steady-state operating point
    for _ in range(dfIMC.median_window_size + 1):
        dfIMC.filter(y0)
        dfPID.filter(y0)

    # 2. Pre-fill controller history explicitly
    controllerIMC.set_auto(False)
    controllerPID.set_auto(False)
    for _ in range(3):
        controllerIMC.update_memory(u=u0, e=0.0)
        controllerPID.update_memory(u=u0, y=y0, r=y0)
    controllerIMC.set_auto(True)
    controllerPID.set_auto(True)

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

    # IMC Data Containers
    ymeasIMC_arr = np.zeros(time_steps)
    yhatIMC_arr = np.zeros(time_steps)
    uIMC_arr = np.zeros(time_steps)
    eIMC_arr = np.zeros(time_steps)

    # PID Data Containers
    ymeasPID_arr = np.zeros(time_steps)
    yhatPID_arr = np.zeros(time_steps)
    uPID_arr = np.zeros(time_steps)
    ePID_arr = np.zeros(time_steps)

    # Initial IMC State Iteration
    yIMC = y0
    uIMC = u0

    # Initial PID State Iteration
    yPID = y0
    uPID = u0

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
            yIMC = plantIMC.sample(uIMC)
            yPID = plantPID.sample(uPID)

        # 2. Add disturbance
        if dist0_start_k <= k:
            yIMC += dist0_mag
            yPID += dist0_mag
        if dist1_start_k <= k:
            yIMC += dist1_mag
            yPID += dist1_mag

        # 3. Sensor measurement with additive Gaussian noise
        ymeasIMC = yIMC + np.random.normal(0, noise_std)
        ymeasPID = yPID + np.random.normal(0, noise_std)

        # 4. Filtering pipeline
        yhatIMC = dfIMC.filter(ymeasIMC)
        yhatPID = dfPID.filter(ymeasPID)

        # 5. Error computation
        eIMC = r - yhatIMC
        ePID = r - yhatPID

        # 6. Controller output
        uIMC = controllerIMC.compute(eIMC)
        uPID = controllerPID.compute(r=r, y=yhatPID)

        # Log Data
        r_arr[k] = r

        # IMC Data Logging
        ymeasIMC_arr[k] = ymeasIMC
        yhatIMC_arr[k] = yhatIMC
        uIMC_arr[k] = uIMC
        eIMC_arr[k] = eIMC

        # PID Data Logging
        ymeasPID_arr[k] = ymeasPID
        yhatPID_arr[k] = yhatPID
        uPID_arr[k] = uPID
        ePID_arr[k] = ePID

    # -------------------------------------------------------------------------
    # 3. Plotting Results
    # -------------------------------------------------------------------------

    t = np.arange(time_steps) * Ts

    plt.figure()

    # Output Plot
    plt.subplot(2, 1, 1)
    plt.plot(t, yhatIMC_arr, color="orange", label="Temperatura Estimada (IMC)")
    plt.plot(t, yhatPID_arr, color="blue", label="Temperatura Estimada (PID)")
    plt.plot(t, r_arr, color="red", label="Referência")
    plt.title("Saída do Sistema em Malha Fechada")
    plt.ylabel("Temperatura (°C)")
    plt.xlabel("Tempo (s)")
    plt.legend()

    # Control Signal Plot
    plt.subplot(2, 1, 2)
    plt.step(t, uIMC_arr, color="green", label="Esforço de Controle (IMC)")
    plt.step(t, uPID_arr, color="purple", label="Esforço de Controle (PID)")
    plt.title("Saída do Controlador Digital")
    plt.ylabel("Tensão (V)")
    plt.xlabel("Tempo (s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resposta_malha_fechada_imcpi.png")

    # -------------------------------------------------------------------------
    # 5. Closed-Loop Poles and Zeros
    # -------------------------------------------------------------------------
    G = plantIMC.get_tf()
    F = dfIMC.get_tf()
    C = controllerIMC.get_tf()

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
