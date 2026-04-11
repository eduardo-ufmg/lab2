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


class DigitalControllerPI:
    """
    Digital PI controller with anti-windup. The difference equation is implemented
    directly in terms of the proportional gain Kp and integral gain Ki.
    """

    def __init__(
        self, Kp: float, Ti: float, T_s: float, u_min: float = 2.0, u_max: float = 8.0
    ):

        self.Ts = T_s

        self.u_min = u_min
        self.u_max = u_max

        # Proportional and integral gains
        self.Kp = Kp
        self.Ki = Kp / Ti

        # Past error states: e[n-1]
        self.e_1 = 0.0

        # Past control state: u[n-1]
        self.u_1 = 0.0

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
        self.e_1 = e
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

        du_p = self.Kp * (e - self.e_1)  # Proportional term based on error change
        du_i = self.Ki * e * self.Ts  # Integral term based on current error

        v_new = self.u_1 + du_p + du_i  # Incremental control signal

        # Saturation
        u_new = max(self.u_min, min(self.u_max, v_new))

        # Anti-windup: If saturation occurred, adjust the integral term to prevent windup
        saturation_high = v_new > self.u_max
        saturation_low = v_new < self.u_min

        if (saturation_high and e > 0) or (saturation_low and e < 0):
            v_new = (
                self.u_1 + du_p
            )  # Remove integral contribution if it would cause windup
            u_new = max(self.u_min, min(self.u_max, v_new))

        # Update states with the saturated value
        self.e_1 = e
        self.u_1 = u_new

        return u_new

    def get_tf(self) -> control.TransferFunction:
        """Return the transfer function representation of the digital PI controller."""
        # Discrete-time PI controller transfer function: C(z) = Kp + Ki * Ts / (z - 1)
        num = [self.Kp + self.Ki * self.Ts, -self.Kp]
        den = [1.0, -1.0]
        return control.TransferFunction(num, den, dt=self.Ts)
