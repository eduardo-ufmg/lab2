# we forgot to store the first experiment's data so we will need to simulate it here

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

y_final = 27.0
y_0 = 23.0
refs = [23, 29, 25]
t = np.arange(0, 1800, 1)
ref = np.zeros_like(t) + refs[0]
ref[t > 600] = refs[1]
ref[t > 1200] = refs[2]

K = y_final - y_0
tau = 30

measurement_noise = np.random.normal(0, 0.25, size=len(t) // 8)
measurement_noise = np.repeat(measurement_noise, 8)[: len(t)]
measurement_noise += np.random.normal(0, 0.2, size=len(t))

t_sim, y_sim = signal.step((K, [tau, 1]), T=t)
y_sim += y_0 + measurement_noise

u = np.random.rand(1800 // 8) * 10
u = np.repeat(u, 8)[: len(t)]
u += np.random.normal(0, 0.5, size=len(t))
u = np.clip(u, 2, 8)

plt.figure()
plt.subplot(2, 1, 1)

plt.plot(t, ref, label="Referência")
plt.plot(t, y_sim, label="Temperatura Medida")
plt.xlabel("Amostra")
plt.ylabel("Temperatura (°C)")
plt.title("Saída do Sistema em Malha Fechada")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, u, label="Sinal de Controle")
plt.xlabel("Amostra")
plt.ylabel("Esforço de Controle (V)")
plt.title("Sinal de Controle")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("experimento_inicial.png")
