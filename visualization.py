import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.widgets import Slider, Button, RadioButtons

from pendulum_physics import InvertedPendulum


class ContinuousSimulation:
    def __init__(self):
        self.pendulum = InvertedPendulum()
        self.controller_type = "PID"
        self.dt = 0.02
        self.t = 0.0

        self.max_time_log = 2000
        self.time_log = []
        self.x_log = []
        self.theta_log = []
        self.control_log = []
        self.disturb_log = []

        self.fig = plt.figure(figsize=(14, 12))

        gs = plt.GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.3)

        self.ax_pend = self.fig.add_subplot(gs[0, :], xlim=(-3, 3), ylim=(-1, 1.5))
        self.ax_pend.set_aspect('equal')
        self.ax_pend.grid(True)
        self.ax_pend.set_title("Inverted Pendulum Simulation", fontsize=12, pad=10)

        # Cart and pendulum
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.cart = Rectangle(
            (-self.cart_width / 2, -self.cart_height / 2),
            self.cart_width,
            self.cart_height,
            fc='yellow',
            ec='black',
            lw=2
        )
        self.ax_pend.add_patch(self.cart)
        self.left_wheel = Circle(
            (-0.1, -self.cart_height / 2), 0.05, fc='green', ec='black'
        )
        self.right_wheel = Circle(
            (0.1, -self.cart_height / 2), 0.05, fc='green', ec='black'
        )
        self.ax_pend.add_patch(self.left_wheel)
        self.ax_pend.add_patch(self.right_wheel)
        self.rod, = self.ax_pend.plot([], [], 'k-', lw=4)
        self.bob = Circle((0, 0), 0.07, fc='red', ec='black', lw=2)
        self.ax_pend.add_patch(self.bob)

        self.control_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->', color='cyan', lw=3, mutation_scale=15
        )
        self.ax_pend.add_patch(self.control_arrow)
        self.disturb_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->', color='magenta', lw=3, mutation_scale=15
        )
        self.ax_pend.add_patch(self.disturb_arrow)
        self.ax_pend.plot(
            [-3, 3],
            [-self.cart_height / 2 - 0.05, -self.cart_height / 2 - 0.05],
            'brown',
            lw=3
        )
        self.ax_pend.text(
            1.8,
            1.2,
            "Cyan = Control\nMagenta = Disturbance",
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7)
        )

        self.info_text = self.ax_pend.text(
            -2.9, 0.8, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.7)
        )

        self.ax_angle = self.fig.add_subplot(gs[1, 0])
        self.ax_position = self.fig.add_subplot(gs[2, 0], sharex=self.ax_angle)
        self.ax_control = self.fig.add_subplot(gs[3, 0], sharex=self.ax_angle)

        plt.setp(self.ax_angle.get_xticklabels(), visible=False)
        plt.setp(self.ax_position.get_xticklabels(), visible=False)

        self.ax_angle.set_title("Angle Response", fontsize=11)
        self.ax_angle.set_ylabel("Angle (deg)")
        self.ax_angle.grid(True)
        self.angle_line, = self.ax_angle.plot([], [], 'r-')

        self.ax_position.set_title("Cart Position", fontsize=11)
        self.ax_position.set_ylabel("Position (m)")
        self.ax_position.grid(True)
        self.position_line, = self.ax_position.plot([], [], 'b-')

        self.ax_control.set_title("Control Force", fontsize=11)
        self.ax_control.set_xlabel("Time (s)")
        self.ax_control.set_ylabel("Control (N)")
        self.ax_control.grid(True)
        self.control_line, = self.ax_control.plot([], [], 'g-')

        self.ax_dist = self.fig.add_subplot(gs[1:, 1])
        self.ax_dist.set_title("Disturbance Force", fontsize=11)
        self.ax_dist.set_xlabel("Time (s)")
        self.ax_dist.set_ylabel("Force (N)")
        self.ax_dist.grid(True)
        self.disturb_line, = self.ax_dist.plot([], [], 'm-', label='Disturbance (N)')
        self.ax_dist.legend(loc='upper right')

        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.95)

        self.add_widgets()

        self.anim = FuncAnimation(
            self.fig, self.animate, init_func=self.init_anim, interval=20, blit=True
        )

    def add_widgets(self):
        control_ax = plt.axes([0.05, 0.03, 0.9, 0.18], facecolor='lightgray')
        control_ax.set_title("Controls", fontsize=10)
        control_ax.set_xticks([])
        control_ax.set_yticks([])

        # PID sliders
        ax_kp = plt.axes([0.25, 0.15, 0.65, 0.02])
        ax_ki = plt.axes([0.25, 0.11, 0.65, 0.02])
        ax_kd = plt.axes([0.25, 0.07, 0.65, 0.02])
        self.slider_kp = Slider(
            ax_kp, 'Kp', 0, 100,
            valinit=self.pendulum.pid_controller.pid_gains['Kp']
        )
        self.slider_ki = Slider(
            ax_ki, 'Ki', 0, 10,
            valinit=self.pendulum.pid_controller.pid_gains['Ki']
        )
        self.slider_kd = Slider(
            ax_kd, 'Kd', 0, 20,
            valinit=self.pendulum.pid_controller.pid_gains['Kd']
        )
        self.slider_kp.on_changed(self.update_pid)
        self.slider_ki.on_changed(self.update_pid)
        self.slider_kd.on_changed(self.update_pid)

        # Impulse slider
        ax_impulse = plt.axes([0.25, 0.19, 0.65, 0.02])
        self.slider_impulse = Slider(ax_impulse, 'Impulse', -5, 5, valinit=0)
        self.slider_impulse.on_changed(self.trigger_impulse)

        # Controller radio buttons
        ax_radio = plt.axes([0.07, 0.08, 0.13, 0.12])
        self.radio_controller = RadioButtons(
            ax_radio, ['PID', 'Pole', 'Nonlinear'], active=0
        )
        self.radio_controller.on_clicked(self.update_controller)

        button_w = 0.09
        ax_noise = plt.axes([0.07, 0.02, button_w, 0.04])
        self.button_noise = Button(ax_noise, 'Noise: OFF')
        self.button_noise.on_clicked(self.toggle_noise)

        ax_filter = plt.axes([0.07 + button_w + 0.01, 0.02, button_w, 0.04])
        self.button_filter = Button(ax_filter, 'Filter: OFF')
        self.button_filter.on_clicked(self.toggle_filter)

        ax_reset = plt.axes([0.07 + 2 * button_w + 0.02, 0.02, button_w, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_simulation)

        ax_run_all = plt.axes([0.07 + 3 * button_w + 0.03, 0.02, button_w + 0.04, 0.04])
        self.button_run_all = Button(ax_run_all, 'Run All Tests')
        self.button_run_all.on_clicked(self.run_all_comparisons)

    def update_pid(self, val=None):
        self.pendulum.pid_controller.pid_gains['Kp'] = self.slider_kp.val
        self.pendulum.pid_controller.pid_gains['Ki'] = self.slider_ki.val
        self.pendulum.pid_controller.pid_gains['Kd'] = self.slider_kd.val

    def trigger_impulse(self, val):
        impulse_val = self.slider_impulse.val
        if abs(impulse_val) > 1e-3:
            self.pendulum.impulse_disturbance = impulse_val
            self.pendulum.impulse_frames_remaining = 5
            self.slider_impulse.set_val(0)

    def update_controller(self, label):
        if label == 'PID':
            self.controller_type = "PID"
        elif label == 'Pole':
            self.controller_type = "pole_placement"
        elif label == 'Nonlinear':
            self.controller_type = "nonlinear"

    def toggle_noise(self, event=None):
        self.pendulum.noise_enabled = not self.pendulum.noise_enabled
        self.button_noise.label.set_text(
            f"Noise: {'ON' if self.pendulum.noise_enabled else 'OFF'}"
        )

    def toggle_filter(self, event=None):
        self.pendulum.filter_enabled = not self.pendulum.filter_enabled
        self.pendulum.filter_history = []
        self.button_filter.label.set_text(
            f"Filter: {'ON' if self.pendulum.filter_enabled else 'OFF'}"
        )

    def reset_simulation(self, event=None):
        self.t = 0.0
        self.pendulum.state = np.array([0.0, 0.0, 0.04, 0.0])
        self.pendulum.pid_controller.reset()
        self.pendulum.pole_controller.reset()
        self.pendulum.nonlinear_controller.reset()
        self.pendulum.filter_history = []
        self.time_log.clear()
        self.x_log.clear()
        self.theta_log.clear()
        self.control_log.clear()
        self.disturb_log.clear()
        self.slider_impulse.set_val(0)

    def run_performance_test(
        self, controller_types=["PID", "pole_placement", "nonlinear"],
        disturbance=3.0, noise=False, filter=False, test_time=10.0
    ):
        results = {}

        if not hasattr(self, 'status_text'):
            self.status_text = self.ax_pend.text(
                0, -0.8, "", fontsize=12, color='blue',
                bbox=dict(facecolor='white', alpha=0.8), ha='center'
            )

        for controller in controller_types:
            self.status_text.set_text(f"Testing {controller}...\nPlease wait.")
            self.fig.canvas.draw_idle()
            orig_controller = self.controller_type
            orig_noise = self.pendulum.noise_enabled
            orig_filter = self.pendulum.filter_enabled

            self.reset_simulation()
            self.controller_type = controller
            self.pendulum.noise_enabled = noise
            self.pendulum.filter_enabled = filter

            self.pendulum.impulse_disturbance = disturbance
            self.pendulum.impulse_frames_remaining = 5

            time_points = []
            angle_points = []
            position_points = []
            control_points = []

            test_steps = int(test_time / self.dt)
            for step in range(test_steps):
                self.pendulum.step_dynamics(self.dt, controller)
                x, x_dot, theta, theta_dot = self.pendulum.state

                x_meas, x_dot_meas = x, x_dot
                theta_meas, theta_dot_meas = theta, theta_dot

                if self.pendulum.noise_enabled:
                    x_meas += np.random.normal(0, 0.01)
                    x_dot_meas += np.random.normal(0, 0.02)
                    theta_meas += np.random.normal(0, 0.01)
                    theta_dot_meas += np.random.normal(0, 0.02)

                if self.pendulum.filter_enabled:
                    self.pendulum.filter_history.append(
                        [x_meas, x_dot_meas, theta_meas, theta_dot_meas]
                    )
                    if len(self.pendulum.filter_history) > self.pendulum.filter_len:
                        self.pendulum.filter_history.pop(0)

                    if len(self.pendulum.filter_history) >= 3:
                        weights = np.linspace(
                            0.5, 1.0, len(self.pendulum.filter_history)
                        )
                        weights = weights / np.sum(weights)

                        x_meas = 0
                        x_dot_meas = 0
                        theta_meas = 0
                        theta_dot_meas = 0

                        for i, hist in enumerate(self.pendulum.filter_history):
                            x_meas += hist[0] * weights[i]
                            x_dot_meas += hist[1] * weights[i]
                            theta_meas += hist[2] * weights[i]
                            theta_dot_meas += hist[3] * weights[i]

                if controller == "PID":
                    ctrl = self.pendulum.pid_controller.compute_control(
                        theta_meas, theta_dot_meas, self.dt, x_meas
                    )
                elif controller == "pole_placement":
                    ctrl = self.pendulum.pole_controller.compute_control(
                        theta_meas, theta_dot_meas
                    )
                elif controller == "nonlinear":
                    ctrl = self.pendulum.nonlinear_controller.compute_control(
                        theta_meas, theta_dot_meas,
                        self.pendulum.m, self.pendulum.L, self.pendulum.g
                    )
                else:
                    ctrl = 0.0

                ctrl = np.clip(ctrl, -20, 20)

                time_points.append(step * self.dt)
                angle_points.append(np.degrees(theta))
                position_points.append(x)
                control_points.append(ctrl)

            response_time = self.calculate_response_time(time_points, angle_points)
            settling_time = self.calculate_settling_time(time_points, angle_points)
            max_deviation = np.max(np.abs(np.array(angle_points)))
            control_effort = np.sum(np.abs(np.array(control_points))) * self.dt

            results[controller] = {
                'time': time_points,
                'angle': angle_points,
                'position': position_points,
                'control': control_points,
                'response_time': response_time,
                'settling_time': settling_time,
                'max_deviation': max_deviation,
                'control_effort': control_effort
            }

            self.controller_type = orig_controller
            self.pendulum.noise_enabled = orig_noise
            self.pendulum.filter_enabled = orig_filter

        return results

    def calculate_response_time(self, time, angle):
        max_angle = max(abs(max(angle)), abs(min(angle)))
        recovery_threshold = max_angle * 0.25
        peak_idx = np.argmax(np.abs(angle))
        for i in range(peak_idx, len(time) - 1):
            if abs(angle[i]) <= recovery_threshold:
                t0, t1 = time[i - 1], time[i]
                a0, a1 = abs(angle[i - 1]), abs(angle[i])
                if a0 != a1:
                    t_cross = t0 + (t1 - t0) * (recovery_threshold - a0) / (a1 - a0)
                    return t_cross - time[peak_idx]
                return time[i] - time[peak_idx]
        return time[-1] - time[peak_idx]

    def calculate_settling_time(self, time, angle):
        threshold = 0.5
        window = 50
        start_idx = 0
        for i in range(len(angle)):
            if abs(angle[i]) > threshold:
                start_idx = i
                break
        for i in range(start_idx, len(time) - window):
            if all(abs(a) <= threshold for a in angle[i:i + window]):
                return time[i] - time[start_idx]
        return time[-1] - time[start_idx]

    def run_all_comparisons(self, event=None):
        if not hasattr(self, 'status_text'):
            self.status_text = self.ax_pend.text(
                0,
                -0.8,
                "",
                fontsize=12,
                color='blue',
                bbox=dict(facecolor='white', alpha=0.8),
                ha='center'
            )
        self.status_text.set_text("Running all comparison scenarios...\nPlease wait.")
        self.fig.canvas.draw_idle()
        results1 = self.run_performance_test(disturbance=3.0, noise=True, filter=True)
        results2 = self.run_performance_test(disturbance=3.0, noise=False, filter=False)
        results3 = self.run_performance_test(disturbance=3.0, noise=True, filter=False)

        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        for controller, data in results1.items():
            plt.plot(data['time'], data['angle'], label=controller)
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Angle Response to Disturbance')
        plt.legend()

        plt.subplot(2, 2, 2)
        for controller, data in results1.items():
            plt.plot(data['time'], data['control'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.title('Control Effort')
        plt.legend()

        plt.subplot(2, 2, 3)
        for controller, data in results1.items():
            plt.plot(data['time'], data['position'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.title('Cart Position')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics = [
            ['Controller', 'Response Time (s)', 'Settling Time (s)',
             'Max Deviation (°)', 'Control Effort (N·s)']
        ]
        for controller, data in results1.items():
            metrics.append([
                controller,
                f"{data['response_time']:.3f}",
                f"{data['settling_time']:.2f}",
                f"{data['max_deviation']:.2f}",
                f"{data['control_effort']:.2f}"
            ])
        table = plt.table(cellText=metrics, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Performance Metrics (Noise: ON, Filter: ON)')
        plt.suptitle('Controller Comparison - Noise ON, Filter ON', fontsize=16)
        plt.tight_layout()
        plt.savefig('scenario1_noise_on_filter_on.png', dpi=300)

        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        for controller, data in results2.items():
            plt.plot(data['time'], data['angle'], label=controller)
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Angle Response to Disturbance')
        plt.legend()

        plt.subplot(2, 2, 2)
        for controller, data in results2.items():
            plt.plot(data['time'], data['control'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.title('Control Effort')
        plt.legend()

        plt.subplot(2, 2, 3)
        for controller, data in results2.items():
            plt.plot(data['time'], data['position'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.title('Cart Position')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics = [
            ['Controller', 'Response Time (s)', 'Settling Time (s)',
             'Max Deviation (°)', 'Control Effort (N·s)']
        ]
        for controller, data in results2.items():
            metrics.append([
                controller,
                f"{data['response_time']:.3f}",
                f"{data['settling_time']:.2f}",
                f"{data['max_deviation']:.2f}",
                f"{data['control_effort']:.2f}"
            ])
        table = plt.table(cellText=metrics, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Performance Metrics (Noise: OFF, Filter: OFF)')
        plt.suptitle('Controller Comparison - Noise OFF, Filter OFF', fontsize=16)
        plt.tight_layout()
        plt.savefig('scenario2_noise_off_filter_off.png', dpi=300)

        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        for controller, data in results3.items():
            plt.plot(data['time'], data['angle'], label=controller)
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Angle Response to Disturbance')
        plt.legend()

        plt.subplot(2, 2, 2)
        for controller, data in results3.items():
            plt.plot(data['time'], data['control'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.title('Control Effort')
        plt.legend()

        plt.subplot(2, 2, 3)
        for controller, data in results3.items():
            plt.plot(data['time'], data['position'], label=controller)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.title('Cart Position')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics = [
            ['Controller', 'Response Time (s)', 'Settling Time (s)',
             'Max Deviation (°)', 'Control Effort (N·s)']
        ]
        for controller, data in results3.items():
            metrics.append([
                controller,
                f"{data['response_time']:.3f}",
                f"{data['settling_time']:.2f}",
                f"{data['max_deviation']:.2f}",
                f"{data['control_effort']:.2f}"
            ])
        table = plt.table(cellText=metrics, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Performance Metrics (Noise: ON, Filter: OFF)')
        plt.suptitle('Controller Comparison - Noise ON, Filter OFF', fontsize=16)
        plt.tight_layout()
        plt.savefig('scenario3_noise_on_filter_off.png', dpi=300)
        plt.show()

        self.status_text.set_text("All comparisons complete.\nResults displayed and saved.")
        self.fig.canvas.draw_idle()

    def init_anim(self):
        self.cart.set_xy((-self.cart_width / 2, -self.cart_height / 2))
        self.left_wheel.center = (-0.1, -self.cart_height / 2)
        self.right_wheel.center = (0.1, -self.cart_height / 2)
        self.rod.set_data([], [])
        self.bob.center = (0, 0)
        self.control_arrow.set_positions((0, 0), (0, 0))
        self.disturb_arrow.set_positions((0, 0), (0, 0))
        self.angle_line.set_data([], [])
        self.position_line.set_data([], [])
        self.control_line.set_data([], [])
        self.disturb_line.set_data([], [])
        self.info_text.set_text("")
        return (
            self.cart,
            self.left_wheel,
            self.right_wheel,
            self.rod,
            self.bob,
            self.control_arrow,
            self.disturb_arrow,
            self.angle_line,
            self.position_line,
            self.control_line,
            self.disturb_line,
            self.info_text
        )

    def animate(self, frame):
        self.pendulum.step_dynamics(self.dt, self.controller_type)
        self.t += self.dt

        x, x_dot, theta, theta_dot = self.pendulum.state

        x_meas, x_dot_meas = x, x_dot
        theta_meas, theta_dot_meas = theta, theta_dot

        if self.pendulum.noise_enabled:
            x_meas += np.random.normal(0, 0.01)
            x_dot_meas += np.random.normal(0, 0.02)
            theta_meas += np.random.normal(0, 0.01)
            theta_dot_meas += np.random.normal(0, 0.02)

        if self.pendulum.filter_enabled:
            self.pendulum.filter_history.append(
                [x_meas, x_dot_meas, theta_meas, theta_dot_meas]
            )
            if len(self.pendulum.filter_history) > self.pendulum.filter_len:
                self.pendulum.filter_history.pop(0)

            if len(self.pendulum.filter_history) >= 3:
                weights = np.linspace(
                    0.5, 1.0, len(self.pendulum.filter_history)
                )
                weights = weights / np.sum(weights)

                x_meas = 0
                x_dot_meas = 0
                theta_meas = 0
                theta_dot_meas = 0

                for i, hist in enumerate(self.pendulum.filter_history):
                    x_meas += hist[0] * weights[i]
                    x_dot_meas += hist[1] * weights[i]
                    theta_meas += hist[2] * weights[i]
                    theta_dot_meas += hist[3] * weights[i]

        if self.controller_type == "PID":
            ctrl = self.pendulum.pid_controller.compute_control(
                theta_meas, theta_dot_meas, self.dt, x_meas
            )
        elif self.controller_type == "pole_placement":
            ctrl = self.pendulum.pole_controller.compute_control(
                theta_meas, theta_dot_meas
            )
        elif self.controller_type == "nonlinear":
            ctrl = self.pendulum.nonlinear_controller.compute_control(
                theta_meas, theta_dot_meas,
                self.pendulum.m, self.pendulum.L, self.pendulum.g
            )
        else:
            ctrl = 0.0

        ctrl = np.clip(ctrl, -20, 20)
        dist_force = 0.0
        if self.pendulum.impulse_frames_remaining > 0:
            dist_force = self.pendulum.impulse_disturbance
        total_dist = dist_force + self.pendulum.constant_disturbance

        self.time_log.append(self.t)
        self.x_log.append(x)
        self.theta_log.append(theta)
        self.control_log.append(ctrl)
        self.disturb_log.append(total_dist)
        if len(self.time_log) > self.max_time_log:
            self.time_log.pop(0)
            self.x_log.pop(0)
            self.theta_log.pop(0)
            self.control_log.pop(0)
            self.disturb_log.pop(0)

        self.cart.set_xy((x - self.cart_width / 2, -self.cart_height / 2))
        self.left_wheel.center = (x - 0.1, -self.cart_height / 2)
        self.right_wheel.center = (x + 0.1, -self.cart_height / 2)

        L = self.pendulum.L
        rod_x = x
        rod_y = 0
        bob_x = rod_x + L * np.sin(theta)
        bob_y = rod_y + L * np.cos(theta)
        self.rod.set_data([rod_x, bob_x], [rod_y, bob_y])
        self.bob.center = (bob_x, bob_y)

        arrow_scale = 0.03
        self.control_arrow.set_positions((x, -0.05), (x + arrow_scale * ctrl, -0.05))
        if abs(total_dist) > 0.01:
            self.disturb_arrow.set_positions(
                (x, 0.1), (x + arrow_scale * total_dist, 0.1)
            )
            self.disturb_arrow.set_visible(True)
        else:
            self.disturb_arrow.set_visible(False)

        angle_deg = np.degrees(theta)
        info_str = f"Controller = {self.controller_type}\nAngle = {angle_deg:.1f}°\n"
        self.info_text.set_text(info_str)

        t_arr = np.array(self.time_log)
        x_arr = np.array(self.x_log)
        th_arr_deg = np.degrees(np.array(self.theta_log))
        ctrl_arr = np.array(self.control_log)
        dist_arr = np.array(self.disturb_log)

        self.angle_line.set_data(t_arr, th_arr_deg)
        self.position_line.set_data(t_arr, x_arr)
        self.control_line.set_data(t_arr, ctrl_arr)
        self.disturb_line.set_data(t_arr, dist_arr)

        if len(t_arr) > 1:
            time_min = t_arr[0]
            time_max = max(t_arr[-1], time_min + 1.5)
            self.ax_angle.set_xlim(time_min, time_max)
            self.ax_dist.set_xlim(time_min, time_max)

            if len(th_arr_deg) > 0:
                y_min = min(-2.0, np.min(th_arr_deg))
                y_max = max(2.0, np.max(th_arr_deg))
                margin = 0.2 * (y_max - y_min)
                self.ax_angle.set_ylim(y_min - margin, y_max + margin)

            if len(x_arr) > 0:
                y_min = min(-0.01, np.min(x_arr))
                y_max = max(0.01, np.max(x_arr))
                margin = 0.1 * (y_max - y_min)
                self.ax_position.set_ylim(y_min - margin, y_max + margin)

            if len(ctrl_arr) > 0:
                y_min = min(-0.1, np.min(ctrl_arr))
                y_max = max(0.1, np.max(ctrl_arr))
                margin = 0.1 * (y_max - y_min)
                self.ax_control.set_ylim(y_min - margin, y_max + margin)

            if len(dist_arr) > 0:
                y_range = max(0.1, np.max(dist_arr) - np.min(dist_arr))
                y_mid = (np.max(dist_arr) + np.min(dist_arr)) / 2
                self.ax_dist.set_ylim(y_mid - y_range, y_mid + y_range)

        return (
            self.cart,
            self.left_wheel,
            self.right_wheel,
            self.rod,
            self.bob,
            self.control_arrow,
            self.disturb_arrow,
            self.angle_line,
            self.position_line,
            self.control_line,
            self.disturb_line,
            self.info_text
        )

    def main_loop(self):
        plt.show()
