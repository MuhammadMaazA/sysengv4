import numpy as np
from scipy import signal

from controllers import PIDController, PoleController, NonlinearController


class InvertedPendulum:
    def __init__(self):
        #system parameters
        self.m = 0.1       #pendulum bob mass
        self.M = 0.5       #cart mass
        self.L = 0.5       #pendulum length
        self.g = 9.81      #gravity

        # friction parameters
        self.b = 0.5       #baseline cart damping
        self.bd = 0.1      #  pivot damping
        self.cart_friction = 4  # cart friction
        self.state = np.array([0.0, 0.0, 0.04, 0.0])

        self.pid_controller = PIDController({'Kp': 1.2, 'Ki': 1.63, 'Kd': 0.22})
        self.pole_controller = PoleController(self.angle_only_pole_placement())
        self.nonlinear_controller = NonlinearController({'k2': 1.0})
        self.constant_disturbance = 0.0
        self.impulse_disturbance = 0.0
        self.impulse_frames_remaining = 0
        self.noise_enabled = False
        self.filter_enabled = False
        self.filter_history = []
        self.filter_len = 5

    def angle_only_pole_placement(self):
        A2 = np.array([
            [0, 1],
            [self.g / self.L, 0]
        ])
        B2 = np.array([
            [0],
            [1 / (self.m * self.L**2)]
        ])
        poles2 = [-3, -4]
        K2 = signal.place_poles(A2, B2, poles2).gain_matrix.flatten()
        return K2

    def step_dynamics(self, dt, controller_type="PID"):
        s0 = self.state
        k1 = self.dynamics(s0, controller_type)
        k2 = self.dynamics(s0 + 0.5 * dt * k1, controller_type)
        k3 = self.dynamics(s0 + 0.5 * dt * k2, controller_type)
        k4 = self.dynamics(s0 + dt * k3, controller_type)
        new_state = s0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x, x_dot, theta, theta_dot = new_state

        # Cart clamp in [-2.5, 2.5]
        # if x < -2.5:
        #    x = -2.5
        # elif x > 2.5:
        #    x = 2.5

        # 2) Angle clamp in [-90°, 90°]
        if theta < -np.pi / 2:
            theta = -np.pi / 2
        elif theta > np.pi / 2:
            theta = np.pi / 2

        self.state = np.array([x, x_dot, theta, theta_dot])

    def dynamics(self, y, controller_type="PID"):
        x, x_dot, theta, theta_dot = y
        m, M, L, g = self.m, self.M, self.L, self.g

        dist_force = 0.0
        if self.impulse_frames_remaining > 0:
            dist_force = self.impulse_disturbance
            self.impulse_frames_remaining -= 1

        x_meas, x_dot_meas = x, x_dot
        theta_meas, theta_dot_meas = theta, theta_dot

        if self.noise_enabled:
            x_meas += np.random.normal(0, 0.01)
            x_dot_meas += np.random.normal(0, 0.02)
            theta_meas += np.random.normal(0, 0.01)
            theta_dot_meas += np.random.normal(0, 0.02)

        if self.filter_enabled:
            self.filter_history.append([
                x_meas, x_dot_meas, theta_meas, theta_dot_meas
            ])
            if len(self.filter_history) > self.filter_len:
                self.filter_history.pop(0)

            # Apply weighted moving average
            if len(self.filter_history) >= 3:
                weights = np.linspace(0.5, 1.0, len(self.filter_history))
                weights = weights / np.sum(weights)

                x_meas = 0
                x_dot_meas = 0
                theta_meas = 0
                theta_dot_meas = 0

                for i, hist in enumerate(self.filter_history):
                    x_meas += hist[0] * weights[i]
                    x_dot_meas += hist[1] * weights[i]
                    theta_meas += hist[2] * weights[i]
                    theta_dot_meas += hist[3] * weights[i]

        if controller_type == "PID":
            u = self.pid_controller.compute_control(
                theta_meas, theta_dot_meas, 0.02, x_meas
            )
        elif controller_type == "pole_placement":
            u = self.pole_controller.compute_control(theta_meas, theta_dot_meas)
        elif controller_type == "nonlinear":
            u = self.nonlinear_controller.compute_control(
                theta_meas, theta_dot_meas, self.m, self.L, self.g
            )
        else:
            u = 0.0

        u = np.clip(u, -20, 20)
        total_force = u + dist_force + self.constant_disturbance

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        denom = (M + m) - m * cos_th**2
        damping = self.b + self.cart_friction

        # Check if pendulum is at extreme angles (close to ±90°)
        # Add extra damping to cart motion when pendulum has fallen
        angle_extremity = min(1.0, 10.0 * (abs(abs(theta) - np.pi / 2) < 0.1))
        extra_damping = 2.0 * angle_extremity

        gravity_effect = m * sin_th * (L * theta_dot**2 + g * cos_th)
        gravity_effect *= (1.0 - 0.8 * angle_extremity)

        x_ddot = (
            total_force
            + gravity_effect
            - (damping + extra_damping) * x_dot
        ) / denom

        pivot_damp = self.bd * theta_dot
        theta_ddot = (
            -total_force * cos_th
            - m * L * (theta_dot**2) * sin_th * cos_th
            - (M + m) * g * sin_th
            + damping * x_dot * cos_th
        ) / (L * denom)
        theta_ddot -= pivot_damp

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
