import numpy as np


class PIDController:
    def __init__(self, gains):
        self.pid_gains = gains
        self.integrator_error = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integrator_error = 0.0
        self.prev_error = 0.0

    def compute_control(self, theta, theta_dot, dt, x=0.0):
        error = theta

        # Add deadzone - don't accumulate tiny errors
        deadzone = 0.001  # About 0.06 degrees
        if abs(error) < deadzone:
            error = 0

        self.integrator_error += error * dt

        position_bias = 0.02 * x
        if abs(error) < 0.05:  # Only apply when angle is nearly stable
            self.integrator_error -= position_bias * dt

        self.integrator_error = np.clip(self.integrator_error, -5, 5)

        d_error = (error - self.prev_error) / dt
        self.prev_error = error

        p_term = self.pid_gains['Kp'] * error
        i_term = self.pid_gains['Ki'] * self.integrator_error
        d_term = self.pid_gains['Kd'] * d_error

        control = -(p_term + i_term + d_term)
        if abs(theta) < 0.0005 and abs(theta_dot) < 0.001:
            control *= 0.5  # Reduce control when very close to equilibrium

        return control


class PoleController:
    def __init__(self, pole_gains):
        self.K_pole = pole_gains

    def reset(self):
        pass

    def compute_control(self, theta, theta_dot):
        k1, k2 = self.K_pole
        ctrl = k1 * theta + k2 * theta_dot

        if abs(ctrl) < 0.1 and abs(theta) > 0.01:
            ctrl *= 2.0

        return ctrl


class NonlinearController:
    def __init__(self, gains):
        self.nl_gains = gains

    def reset(self):
        pass

    def compute_control(self, theta, theta_dot, m, L, g):
        E_desired = m * g * L
        E_current = (
            0.5 * m * L**2 * (theta_dot**2)
            + m * g * L * (1 - np.cos(theta))
        )
        k2 = self.nl_gains['k2']

        ctrl = k2 * (E_desired - E_current) * np.sign(theta_dot * np.cos(theta))

        if abs(ctrl) < 0.1 and abs(theta) > 0.01:
            ctrl *= 1.5

        return ctrl
