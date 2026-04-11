import math
import warnings

import control


class DigitalControllerIMC:
    """
    Digital controller parameterized by the time constant tau and sampling period T_s.
    Implements the difference equation as a function of p = exp(-2*T_s/tau)
    and output saturation with anti-windup.
    """

    def __init__(
        self, tau_cl: float, T_s: float, u_min: float = 2.0, u_max: float = 8.0
    ):

        self.Ts = T_s

        self.u_min = u_min
        self.u_max = u_max

        # Calculate parameter alpha for a critically damped second-order target
        alpha = math.exp(-T_s / tau_cl)
        alpha_sq = alpha * alpha
        one_minus_alpha_sq = (1.0 - alpha) ** 2

        # Numerator coefficients
        self.b0 = -250.0 * one_minus_alpha_sq
        self.b1 = 469.25 * one_minus_alpha_sq
        self.b2 = -219.34 * one_minus_alpha_sq

        # Denominator coefficients
        self.a1 = 2.0 * alpha + 0.88
        self.a2 = -(alpha_sq + 1.76 * alpha)
        self.a3 = alpha_sq - 0.24 * alpha + 0.12

        self.num = [self.b0, self.b1, self.b2]
        self.den = [1.0, -self.a1, -self.a2, -self.a3]

        # Past error states: e[n-1], e[n-2]
        self.e_1 = 0.0
        self.e_2 = 0.0

        # Past control states: u[n-1], u[n-2], u[n-3]
        self.u_1 = 0.0
        self.u_2 = 0.0
        self.u_3 = 0.0

        self.auto = False

    def set_auto(self, auto: bool) -> None:
        """Set the controller mode to automatic or manual."""
        self.auto = auto

    def update_memory(self, u: float, e: float) -> None:
        """
        Manually update the historical states. Intended for manual mode operation
        to ensure bumpless transfer when switching to auto.
        """
        if self.auto:
            warnings.warn(
                "Controller is in auto mode. Memory will not be manually updated."
            )
            return

        # Shift states chronologically
        self.e_2 = self.e_1
        self.e_1 = e

        self.u_3 = self.u_2
        self.u_2 = self.u_1
        self.u_1 = u

    def compute(self, e: float) -> float:
        """
        Compute the saturated control signal u[n] given the current error e[n].
        """
        if not self.auto:
            warnings.warn(
                "Controller is in manual mode. compute() returns last control signal without update."
            )
            return self.u_1  # Return last control signal for consistency in manual mode

        # Parameterized difference equation
        v_new = (
            (self.b0 * e)
            + (self.b1 * self.e_1)
            + (self.b2 * self.e_2)
            + (self.a1 * self.u_1)
            + (self.a2 * self.u_2)
            + (self.a3 * self.u_3)
        )

        # Saturation
        u_new = max(self.u_min, min(self.u_max, v_new))

        # Update states with the saturated value (anti-windup via recalculation)
        self.e_2 = self.e_1
        self.e_1 = e

        self.u_3 = self.u_2
        self.u_2 = self.u_1
        self.u_1 = u_new

        return u_new

    def get_tf(self) -> control.TransferFunction:
        """Return the transfer function representation of the digital controller."""
        return control.TransferFunction(self.num, self.den, dt=self.Ts)


class DigitalControllerPID:
    """
    Digital PID controller with a derivative filter and exact back-calculation anti-windup.
    To ensure robustness to step changes in the reference, the derivative action
    is applied to the process variable (y) rather than the error (e).
    """

    def __init__(
        self,
        Kp: float,
        Ti: float,
        Td: float,
        N: float,
        T_s: float,
        u_min: float = 2.0,
        u_max: float = 8.0,
    ):
        self.Ts = T_s
        self.u_min = u_min
        self.u_max = u_max

        # Controller gains
        self.Kp = float(Kp)
        self.Ki = (self.Kp / Ti) if Ti > 0.0 else 0.0
        self.Kd = self.Kp * Td

        # Derivative filter coefficients (Backward Euler: s = (1 - z^-1)/Ts)
        # Filter time constant Tf = Td / N
        Tf = Td / N if N > 0.0 else 0.0
        den = Tf + self.Ts

        if den > 0.0:
            self.a_d = Tf / den
            self.b_d = self.Kd / den
        else:
            self.a_d = 0.0
            self.b_d = 0.0

        # Memory states
        self.I = 0.0  # Integral state
        self.D = 0.0  # Filtered derivative state
        self.y_1 = 0.0  # Past process variable y[n-1]
        self.u_1 = 0.0  # Past control signal u[n-1]

        self.auto = False

    def set_auto(self, auto: bool) -> None:
        """Set the controller mode to automatic or manual."""
        self.auto = auto

    def update_memory(self, u: float, y: float, r: float) -> None:
        """
        Manually update the historical states. Intended for manual mode operation
        to ensure bumpless transfer when switching to auto.
        """
        if self.auto:
            warnings.warn(
                "Controller is in auto mode. Memory will not be manually updated."
            )
            return

        self.y_1 = y
        self.u_1 = u

        # Back-calculate the integral state to align with the manual control signal.
        # Assumes the derivative state D decays to 0 during steady-state manual operation.
        self.D = 0.0
        e = r - y
        P = self.Kp * e
        self.I = u - P

    def compute(self, r: float, y: float) -> float:
        """
        Compute the saturated control signal u[n] given reference r[n] and measurement y[n].
        """
        if not self.auto:
            warnings.warn(
                "Controller is in manual mode. compute() returns the last control signal."
            )
            return self.u_1

        e = r - y

        # Proportional action
        P = self.Kp * e

        # Integral action (Backward Euler formulation)
        I_new = self.I + (self.Ki * self.Ts * e)

        # Derivative action on -y to prevent derivative kick from reference steps
        D_new = (self.a_d * self.D) - (self.b_d * (y - self.y_1))

        # Unconstrained control signal
        v_new = P + I_new + D_new

        # Saturation
        u_new = max(self.u_min, min(self.u_max, v_new))

        # Anti-windup via exact back-calculation of the integral state
        if v_new != u_new:
            I_new = u_new - P - D_new

        # Update memory states for the next iteration
        self.I = I_new
        self.D = D_new
        self.y_1 = y
        self.u_1 = u_new

        return u_new

    def get_tf(self) -> control.TransferFunction:
        """
        Return the transfer function representation of the digital PID controller.
        For linear analysis, this assumes standard ISA parallel form C(z) = U(z)/E(z).
        """
        tf_P = control.TransferFunction([self.Kp], [1.0], dt=self.Ts)

        if self.Ki > 0.0:
            # I(z) = (Ki * Ts * z) / (z - 1)
            tf_I = control.TransferFunction(
                [self.Ki * self.Ts, 0.0], [1.0, -1.0], dt=self.Ts
            )
        else:
            tf_I = control.TransferFunction([0.0], [1.0], dt=self.Ts)

        if self.b_d > 0.0:
            # D(z) = (b_d * z - b_d) / (z - a_d)
            tf_D = control.TransferFunction(
                [self.b_d, -self.b_d], [1.0, -self.a_d], dt=self.Ts
            )
        else:
            tf_D = control.TransferFunction([0.0], [1.0], dt=self.Ts)

        # The control library overrides the addition operator to compute parallel TFs.
        return tf_P + tf_I + tf_D
