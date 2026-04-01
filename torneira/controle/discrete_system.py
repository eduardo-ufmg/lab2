import numpy as np
import matplotlib.pyplot as plt


class DiscreteSystem:
    """
    First-order discrete system implemented using deviation variables to match
    the continuous ZOH equivalent of G(s) = K / (tau*s + 1).
    """

    def __init__(self, k: float, tau: float, t_s: float, u_op: float, y_op: float):
        self.u_op = u_op
        self.y_op = y_op

        # Exact ZOH coefficient calculation
        self.a1 = np.exp(-t_s / tau)
        self.b1 = k * (1.0 - self.a1)

        # State registers (deviation variables)
        self.delta_y_prev = 0.0
        self.delta_u_prev = 0.0

    def sample(self, u: float) -> float:
        """
        Computes the current absolute output y[k], then updates the state registers.
        """
        # Convert absolute input to deviation input
        delta_u = u - self.u_op

        # Compute current output deviation
        delta_y_new = self.a1 * self.delta_y_prev + self.b1 * self.delta_u_prev

        # Update states for the next discrete time step (k+1)
        self.delta_y_prev = delta_y_new
        self.delta_u_prev = delta_u

        # Convert deviation output to absolute output
        y_new = self.y_op + delta_y_new

        return y_new


if __name__ == "__main__":
    # System parameters
    K = -1.4
    tau = 36
    Ts = 0.1

    # Operating point (Initial conditions)
    u0 = 8.0
    y0 = 23.0

    system = DiscreteSystem(k=K, tau=tau, t_s=Ts, u_op=u0, y_op=y0)

    # Simulation parameters
    time_steps = 2000  # Extended to show steady-state convergence
    u_values = np.zeros(time_steps)
    y_values = np.zeros(time_steps)

    u = u0  # Start with the initial input

    for i in range(time_steps):
        # Apply a step change from 8 to 2 at k=100
        if i > 99:
            u = 2.0

        y = system.sample(u)

        u_values[i] = u
        y_values[i] = y

    # Plot the results
    plt.figure()
    plt.plot(y_values)
    plt.title("Step Response")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Amplitude")
    plt.tight_layout()
    plt.show()
