import numpy as np


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
