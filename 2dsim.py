#!/usr/bin/env python3
"""
Continuous Inverted Pendulum Simulation with Live ODE Integration
- Very small impulses so each push only moves the cart slightly.
- Press Left/Right multiple times if you want repeated nudges.
- PID gains can be changed mid-run without re-initializing the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import signal

class InvertedPendulum:
    def __init__(self):
        # System parameters
        self.m = 0.2
        self.M = 0.5
        self.L = 0.5
        self.g = 9.81
        self.b = 0.1   # Cart damping
        self.bd = 0.01 # Pendulum pivot damping

        # State: [x, x_dot, theta, theta_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        # PID gains
        self.pid_gains = {'Kp': 50.0, 'Ki': 0.0, 'Kd': 10.0}
        self.integrator_error = 0.0
        self.prev_error = 0.0

        # Nonlinear gains
        self.nl_gains = {'k1': 0.5, 'k2': 1.0, 'k3': 0.2}

        # Pole-placement gains
        self.K = self.linearize_system()

        # Disturbances
        self.constant_disturbance = 0.0
        self.impulse_disturbance = 0.0
        self.impulse_frames_remaining = 0

        # Noise/filter flags
        self.noise_enabled = False
        self.filter_enabled = False
        self.last_meas = None  # for filtering

    def linearize_system(self):
        """Compute linearization around upright, do pole placement for K."""
        m, M, L, g, b = self.m, self.M, self.L, self.g, self.b
        A = np.array([
            [0,       1,        0,         0],
            [0,   -b/M,   (m*g)/M,         0],
            [0,       0,        0,         1],
            [0, -b/(M*L), (M+m)*g/(M*L), 0]
        ])
        B = np.array([[0], [1/M], [0], [1/(M*L)]])
        poles = [-3, -3.1, -4, -4.1]
        K = signal.place_poles(A, B, poles).gain_matrix.flatten()
        return K

    def step_dynamics(self, dt, controller_type="PID"):
        """RK4 integration for a single time step."""
        s0 = self.state
        k1 = self.dynamics(s0, controller_type)
        k2 = self.dynamics(s0 + 0.5*dt*k1, controller_type)
        k3 = self.dynamics(s0 + 0.5*dt*k2, controller_type)
        k4 = self.dynamics(s0 + dt*k3, controller_type)
        self.state = s0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def dynamics(self, y, controller_type="PID"):
        """
        Nonlinear dynamics with optional impulse disturbance.
        y = [x, x_dot, theta, theta_dot].
        """
        x, x_dot, theta, theta_dot = y
        m, M, L, g, b, bd = self.m, self.M, self.L, self.g, self.b, self.bd

        # Disturbance force
        disturbance_force = 0.0
        if self.impulse_frames_remaining > 0:
            disturbance_force = self.impulse_disturbance
            self.impulse_frames_remaining -= 1

        # "Measured" states (with optional noise)
        x_meas, x_dot_meas = x, x_dot
        theta_meas, theta_dot_meas = theta, theta_dot

        if self.noise_enabled:
            x_meas += np.random.normal(0, 0.01)
            x_dot_meas += np.random.normal(0, 0.02)
            theta_meas += np.random.normal(0, 0.01)
            theta_dot_meas += np.random.normal(0, 0.02)

        # Optional filter
        if self.filter_enabled:
            alpha = 0.7
            if self.last_meas is None:
                self.last_meas = [x_meas, x_dot_meas, theta_meas, theta_dot_meas]
            else:
                x_meas = alpha*x_meas + (1-alpha)*self.last_meas[0]
                x_dot_meas = alpha*x_dot_meas + (1-alpha)*self.last_meas[1]
                theta_meas = alpha*theta_meas + (1-alpha)*self.last_meas[2]
                theta_dot_meas = alpha*theta_dot_meas + (1-alpha)*self.last_meas[3]
            self.last_meas = [x_meas, x_dot_meas, theta_meas, theta_dot_meas]

        # Controller
        if controller_type == "PID":
            u = self.control_pid(theta_meas, 0.02)  # assume dt=0.02
        elif controller_type == "pole_placement":
            y_meas = np.array([x_meas, x_dot_meas, theta_meas, theta_dot_meas])
            u = -np.dot(self.K, y_meas)
        elif controller_type == "nonlinear":
            y_meas = np.array([x_meas, x_dot_meas, theta_meas, theta_dot_meas])
            u = self.control_nonlinear(y_meas)
        else:
            u = 0.0

        u = np.clip(u, -20, 20)
        total_force = u + disturbance_force + self.constant_disturbance

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        denom = (M + m) - m*cos_th**2

        x_ddot = (total_force + m*sin_th*(L*theta_dot**2 + g*cos_th) - b*x_dot) / denom
        theta_ddot = (
            -total_force*cos_th
            - m*L*theta_dot**2*sin_th*cos_th
            - (M+m)*g*sin_th
            + b*x_dot*cos_th
        ) / (L*denom)
        theta_ddot -= bd*theta_dot

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def control_pid(self, theta, dt):
        """PID control on theta=0."""
        error = theta
        self.integrator_error += error*dt
        self.integrator_error = np.clip(self.integrator_error, -5, 5)
        d_error = (error - self.prev_error)/dt
        p_term = self.pid_gains['Kp']*error
        i_term = self.pid_gains['Ki']*self.integrator_error
        d_term = self.pid_gains['Kd']*d_error
        self.prev_error = error
        return -(p_term + i_term + d_term)

    def control_nonlinear(self, y):
        x, x_dot, theta, theta_dot = y
        m, L, g = self.m, self.L, self.g
        E_desired = m*g*L
        E_current = 0.5*m*L**2*theta_dot**2 + m*g*L*(1 - np.cos(theta))
        k1 = self.nl_gains['k1']
        k2 = self.nl_gains['k2']
        k3 = self.nl_gains['k3']
        u = -k1*x - k3*x_dot + k2*(E_desired - E_current)*np.sign(theta_dot*np.cos(theta))
        return u

class ContinuousSimulation:
    def __init__(self):
        self.pendulum = InvertedPendulum()
        self.controller_type = "PID"
        self.dt = 0.02
        self.t = 0.0

        # Data logs
        self.max_time_log = 2000  # store up to 2000 points
        self.time_log = []
        self.x_log = []
        self.theta_log = []
        self.control_log = []
        self.disturb_log = []

        # Figure
        self.fig = plt.figure(figsize=(14, 10))
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])

        # Pendulum axis
        self.ax_pend = self.fig.add_subplot(gs[0, :], xlim=(-2.5, 2.5), ylim=(-1, 1.5))
        self.ax_pend.set_aspect('equal')
        self.ax_pend.grid(True)
        self.ax_pend.set_title("Continuous Inverted Pendulum (Small Impulses)")

        # Cart & pendulum patches
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.cart = Rectangle((-self.cart_width/2, -self.cart_height/2),
                              self.cart_width, self.cart_height,
                              fc='yellow', ec='black', lw=2)
        self.ax_pend.add_patch(self.cart)
        self.left_wheel = Circle((-0.1, -self.cart_height/2), 0.05, fc='green', ec='black')
        self.right_wheel = Circle((0.1, -self.cart_height/2), 0.05, fc='green', ec='black')
        self.ax_pend.add_patch(self.left_wheel)
        self.ax_pend.add_patch(self.right_wheel)
        self.rod, = self.ax_pend.plot([], [], 'k-', lw=4)
        self.bob = Circle((0, 0), 0.07, fc='red', ec='black', lw=2)
        self.ax_pend.add_patch(self.bob)
        self.control_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->', color='cyan', lw=3)
        self.ax_pend.add_patch(self.control_arrow)
        self.disturb_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->', color='magenta', lw=3)
        self.ax_pend.add_patch(self.disturb_arrow)
        self.ax_pend.plot([-2.5, 2.5], [-self.cart_height/2 - 0.05, -self.cart_height/2 - 0.05],
                          'brown', lw=3)
        self.info_text = self.ax_pend.text(-2.4, 1.3, "", fontsize=10)

        # System response plot
        self.ax_plot = self.fig.add_subplot(gs[1, 0])
        self.ax_plot.set_title("System Response")
        self.ax_plot.set_xlabel("Time (s)")
        self.ax_plot.set_ylabel("Value")
        self.ax_plot.grid(True)
        self.position_line, = self.ax_plot.plot([], [], 'b-', label='Position (m)')
        self.angle_line, = self.ax_plot.plot([], [], 'r-', label='Angle (deg)')
        self.control_line, = self.ax_plot.plot([], [], 'g-', label='Control (N)')
        self.ax_plot.legend(loc='upper right')

        # Disturbance plot
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        self.ax_dist.set_title("Disturbance Force")
        self.ax_dist.set_xlabel("Time (s)")
        self.ax_dist.set_ylabel("Force (N)")
        self.ax_dist.grid(True)
        self.disturb_line, = self.ax_dist.plot([], [], 'm-', label='Disturbance (N)')
        self.ax_dist.legend(loc='upper right')

        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=0.3, hspace=0.4)
        self.add_widgets()

        # Animation
        self.anim = FuncAnimation(self.fig, self.animate, init_func=self.init_anim,
                                  interval=20, blit=True)

    def add_widgets(self):
        control_ax = plt.axes([0.05, 0.05, 0.9, 0.1], facecolor='lightgray')
        control_ax.set_title("Controls", fontsize=10)
        control_ax.set_xticks([])
        control_ax.set_yticks([])

        # PID sliders
        ax_kp = plt.axes([0.25, 0.13, 0.65, 0.02])
        ax_ki = plt.axes([0.25, 0.09, 0.65, 0.02])
        ax_kd = plt.axes([0.25, 0.05, 0.65, 0.02])
        self.slider_kp = Slider(ax_kp, 'Kp', 0, 100, valinit=self.pendulum.pid_gains['Kp'])
        self.slider_ki = Slider(ax_ki, 'Ki', 0, 10, valinit=self.pendulum.pid_gains['Ki'])
        self.slider_kd = Slider(ax_kd, 'Kd', 0, 20, valinit=self.pendulum.pid_gains['Kd'])
        self.slider_kp.on_changed(self.update_pid)
        self.slider_ki.on_changed(self.update_pid)
        self.slider_kd.on_changed(self.update_pid)

        # Radio for controller
        ax_radio = plt.axes([0.07, 0.01, 0.13, 0.1])
        self.radio_controller = RadioButtons(ax_radio, ['PID', 'Pole', 'Nonlinear'], active=0)
        self.radio_controller.on_clicked(self.update_controller)

        # Noise, filter toggles
        button_w = 0.09
        ax_noise = plt.axes([0.22, 0.01, button_w, 0.04])
        self.button_noise = Button(ax_noise, 'Noise: OFF')
        self.button_noise.on_clicked(self.toggle_noise)
        ax_filter = plt.axes([0.22 + button_w + 0.01, 0.01, button_w, 0.04])
        self.button_filter = Button(ax_filter, 'Filter: OFF')
        self.button_filter.on_clicked(self.toggle_filter)

        # Reset
        ax_reset = plt.axes([0.22 + 2*button_w + 0.02, 0.01, button_w, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_simulation)

        # Push left/right
        ax_left = plt.axes([0.22 + 3*button_w + 0.03, 0.01, button_w, 0.04])
        self.button_left = Button(ax_left, 'Left')
        self.button_left.on_clicked(self.push_left)
        ax_right = plt.axes([0.22 + 4*button_w + 0.04, 0.01, button_w, 0.04])
        self.button_right = Button(ax_right, 'Right')
        self.button_right.on_clicked(self.push_right)

    def update_pid(self, val=None):
        self.pendulum.pid_gains['Kp'] = self.slider_kp.val
        self.pendulum.pid_gains['Ki'] = self.slider_ki.val
        self.pendulum.pid_gains['Kd'] = self.slider_kd.val

    def update_controller(self, label):
        if label == 'PID':
            self.controller_type = "PID"
        elif label == 'Pole':
            self.controller_type = "pole_placement"
        else:
            self.controller_type = "nonlinear"

    def toggle_noise(self, event=None):
        self.pendulum.noise_enabled = not self.pendulum.noise_enabled
        self.button_noise.label.set_text(f"Noise: {'ON' if self.pendulum.noise_enabled else 'OFF'}")

    def toggle_filter(self, event=None):
        self.pendulum.filter_enabled = not self.pendulum.filter_enabled
        self.button_filter.label.set_text(f"Filter: {'ON' if self.pendulum.filter_enabled else 'OFF'}")

    def reset_simulation(self, event=None):
        self.t = 0.0
        self.pendulum.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.pendulum.integrator_error = 0.0
        self.pendulum.prev_error = 0.0
        self.pendulum.last_meas = None
        self.time_log.clear()
        self.x_log.clear()
        self.theta_log.clear()
        self.control_log.clear()
        self.disturb_log.clear()

    def push_left(self, event=None):
        # Very small impulse, only 5 frames => ~0.1 s
        self.pendulum.impulse_disturbance = -0.3
        self.pendulum.impulse_frames_remaining = 5

    def push_right(self, event=None):
        self.pendulum.impulse_disturbance = 0.3
        self.pendulum.impulse_frames_remaining = 5

    def init_anim(self):
        self.cart.set_xy((-self.cart_width/2, -self.cart_height/2))
        self.left_wheel.center = (-0.1, -self.cart_height/2)
        self.right_wheel.center = (0.1, -self.cart_height/2)
        self.rod.set_data([], [])
        self.bob.center = (0, 0)
        self.control_arrow.set_positions((0, 0), (0, 0))
        self.disturb_arrow.set_positions((0, 0), (0, 0))
        return (self.cart, self.left_wheel, self.right_wheel,
                self.rod, self.bob, self.control_arrow, self.disturb_arrow)

    def animate(self, frame):
        # Step the ODE
        self.pendulum.step_dynamics(self.dt, self.controller_type)
        self.t += self.dt

        # Current state
        x, x_dot, theta, theta_dot = self.pendulum.state

        # Approx control for logging
        if self.controller_type == "PID":
            ctrl = self.pendulum.control_pid(theta, self.dt)
        elif self.controller_type == "pole_placement":
            meas = np.array([x, x_dot, theta, theta_dot])
            ctrl = -np.dot(self.pendulum.K, meas)
        elif self.controller_type == "nonlinear":
            meas = np.array([x, x_dot, theta, theta_dot])
            ctrl = self.pendulum.control_nonlinear(meas)
        else:
            ctrl = 0.0
        ctrl = np.clip(ctrl, -20, 20)

        # Disturbance for logging
        dist_force = 0.0
        if self.pendulum.impulse_frames_remaining > 0:
            dist_force = self.pendulum.impulse_disturbance
        total_dist = dist_force + self.pendulum.constant_disturbance

        # Update logs
        self.time_log.append(self.t)
        self.x_log.append(x)
        self.theta_log.append(theta)
        self.control_log.append(ctrl)
        self.disturb_log.append(total_dist)

        # Trim logs
        if len(self.time_log) > self.max_time_log:
            self.time_log.pop(0)
            self.x_log.pop(0)
            self.theta_log.pop(0)
            self.control_log.pop(0)
            self.disturb_log.pop(0)

        # Update cart-pendulum geometry
        x_clamped = np.clip(x, -2.4, 2.4)
        self.cart.set_xy((x_clamped - self.cart_width/2, -self.cart_height/2))
        self.left_wheel.center = (x_clamped - 0.1, -self.cart_height/2)
        self.right_wheel.center = (x_clamped + 0.1, -self.cart_height/2)

        L = self.pendulum.L
        rod_x = x_clamped
        rod_y = 0
        bob_x = rod_x + L*np.sin(theta)
        bob_y = rod_y + L*np.cos(theta)
        self.rod.set_data([rod_x, bob_x], [rod_y, bob_y])
        self.bob.center = (bob_x, bob_y)

        # Arrows
        arrow_scale = 0.01
        self.control_arrow.set_positions(
            (x_clamped, -0.05),
            (x_clamped + arrow_scale*ctrl, -0.05)
        )
        if abs(total_dist) > 0.01:
            self.disturb_arrow.set_positions(
                (x_clamped, 0.1),
                (x_clamped + arrow_scale*total_dist, 0.1)
            )
            self.disturb_arrow.set_visible(True)
        else:
            self.disturb_arrow.set_visible(False)

        # Info text
        angle_deg = ((theta*180/np.pi) + 180) % 360 - 180
        info_str = (f"x={x_clamped:.2f} m\n"
                    f"theta={angle_deg:.1f}°\n"
                    f"Controller={self.controller_type}")
        self.info_text.set_text(info_str)

        # Update plots
        t_array = np.array(self.time_log)
        x_array = np.array(self.x_log)
        th_array_deg = ((np.array(self.theta_log)*180/np.pi) + 180) % 360 - 180
        c_array = np.array(self.control_log)
        d_array = np.array(self.disturb_log)

        self.position_line.set_data(t_array, x_array)
        self.angle_line.set_data(t_array, th_array_deg)
        self.control_line.set_data(t_array, c_array)

        if len(t_array) > 1:
            self.ax_plot.set_xlim(t_array[0], t_array[-1])
        all_vals = np.concatenate([x_array, th_array_deg, c_array])
        y_min = np.min(all_vals)
        y_max = np.max(all_vals)
        margin = 0.1*(y_max - y_min if y_max != y_min else 1)
        self.ax_plot.set_ylim(y_min - margin, y_max + margin)

        self.disturb_line.set_data(t_array, d_array)
        if len(t_array) > 1:
            self.ax_dist.set_xlim(t_array[0], t_array[-1])
        d_min = np.min(d_array)
        d_max = np.max(d_array)
        d_margin = 0.1*(d_max - d_min if d_max != d_min else 1)
        self.ax_dist.set_ylim(d_min - d_margin, d_max + d_margin)

        return (self.cart, self.left_wheel, self.right_wheel,
                self.rod, self.bob, self.control_arrow, self.disturb_arrow,
                self.position_line, self.angle_line, self.control_line,
                self.disturb_line, self.info_text)

def main():
    print("=== Continuous Inverted Pendulum with Small Pushes ===")
    print(" - Each Left/Right push is 0.3 N for ~0.1s (5 frames).")
    print(" - If you want more push, press multiple times.")
    print(" - Adjust Kp, Ki, Kd, or switch controllers at any time.")
    sim = ContinuousSimulation()
    plt.show()

if __name__ == "__main__":
    main()

