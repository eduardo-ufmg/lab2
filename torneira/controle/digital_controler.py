import warnings
import math


class DigitalController:
    """
    Digital controller parameterized by the time constant tau and sampling period T_s.
    Implements the difference equation as a function of p = exp(-2*T_s/tau)
    and output saturation with anti-windup.
    """

    def __init__(
        self, tau_cl: float, T_s: float, u_min: float = 2.0, u_max: float = 8.0
    ):
        self.u_min = u_min
        self.u_max = u_max

        # Calculate parameter p and analytical coefficients
        p = math.exp(-T_s / tau_cl)

        self.b0 = -250.0 * (1.0 - p)
        self.b1 = 469.25 * (1.0 - p)
        self.b2 = -219.34 * (1.0 - p)

        self.a1 = p + 0.88
        self.a2 = -0.88 * p
        self.a3 = 0.12 * (1.0 - p)

        print(
            f"u[n] = {self.b0:.2f}*e[n] + {self.b1:.2f}*e[n-1] + {self.b2:.2f}*e[n-2] + {self.a1:.2f}*u[n-1] + {self.a2:.2f}*u[n-2] + {self.a3:.2f}*u[n-3]"
        )

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
