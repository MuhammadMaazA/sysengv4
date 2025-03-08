import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-10.0, 10.0), dead_zone=1e-4):
        """
        PID Controller with dead zone option.
        dead_zone: if abs(error) < dead_zone, treat error as 0.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error = 0.0
        self.output_limits = output_limits
        self.dead_zone = dead_zone

    def compute(self, measurement, dt):
        """Compute control output based on measurement."""
        error = measurement - self.setpoint
        if abs(error) < self.dead_zone:
            error = 0.0
        
        # Anti-windup: limit integral term
        if self.Ki != 0:
            self.integral = max(self.output_limits[0]/self.Ki, 
                               min(self.integral + error * dt, 
                                   self.output_limits[1]/self.Ki))
        else:
            self.integral += error * dt
            
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        return output, error

    def reset(self):
        """Reset the controller's internal state."""
        self.integral = 0.0
        self.previous_error = 0.0

class PolePlacementController:
    def __init__(self, A, B, poles, output_limits=(-10.0, 10.0)):
        """
        Pole Placement controller.
        
        Args:
            A: System matrix
            B: Input matrix
            poles: Desired closed-loop poles
            output_limits: Control output limits
        """
        self.A = A
        self.B = B
        self.K = self._compute_gain(poles)
        self.output_limits = output_limits
        
    def _compute_gain(self, poles):
        """Compute the pole placement gain matrix K."""
        # Use scipy's place function to compute the gain matrix K
        from scipy.signal import place_poles
        
        # Ensure poles are in a suitable format
        if isinstance(poles, (list, tuple)):
            poles = np.array(poles)
        
        # Compute the gain matrix K
        result = place_poles(self.A, self.B, poles)
        K = result.gain_matrix
        return K
    
    def compute(self, state_vector):
        """
        Compute control output based on current state.
        
        Args:
            state_vector: Current state [position, velocity, angle, angular_velocity]
        
        Returns:
            control: Control output
        """
        state = np.matrix(state_vector).T
        control = -float(self.K @ state)
        control = max(self.output_limits[0], min(control, self.output_limits[1]))
        return control

class NonlinearController:
    def __init__(self, k1=10.0, k2=15.0, k3=20.0, k4=5.0, output_limits=(-10.0, 10.0)):
        """
        Nonlinear controller for inverted pendulum.
        
        Args:
            k1, k2, k3, k4: Controller gains
            output_limits: Control output limits
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.output_limits = output_limits
        
    def compute(self, state_vector):
        """
        Compute nonlinear control output based on current state.
        
        Args:
            state_vector: Current state [position, velocity, angle, angular_velocity]
        
        Returns:
            control: Control output
        """
        x, x_dot, theta, theta_dot = state_vector
        
        # Nonlinear control law
        control = self.k1 * x + self.k2 * x_dot + self.k3 * np.sign(theta) * np.sqrt(abs(theta)) + self.k4 * theta_dot
        
        # Apply output limits
        control = max(self.output_limits[0], min(control, self.output_limits[1]))
        return control

class InvertedPendulumModel:
    """Mathematical model of the inverted pendulum system."""
    
    def __init__(self, M=1.0, m=0.1, L=0.5, g=9.81, b=0.1, I=0.05, air_drag=0.01):
        """
        Args:
            M: Mass of the cart (kg)
            m: Mass of the pendulum (kg)
            L: Length of the pendulum (m)
            g: Gravity acceleration (m/s^2)
            b: Friction coefficient of the cart
            I: Moment of inertia of the pendulum
            air_drag: Air drag coefficient
        """
        self.M = M
        self.m = m
        self.L = L
        self.g = g
        self.b = b
        self.I = I
        self.air_drag = air_drag
        
    # Around line ~330, update the dynamics method in InvertedPendulumModel class
    def dynamics(self, t, state, u=0.0, noise=None):
        """
        Nonlinear dynamics of the inverted pendulum.
        Args:
        t: Time (not used, but required for ODE solvers)
        state: System state [x, x_dot, theta, theta_dot]
        u: Control input force
        noise: Dictionary of noise values for sensors
    
        Returns:
        Derivatives of state
        """
        x, x_dot, theta, theta_dot = state
    
        # Add sensor noise if provided
        if noise is not None:
            theta += np.random.normal(0, noise.get('angle', 0))
            x += np.random.normal(0, noise.get('position', 0))

        # Enhanced nonlinear dynamics with more pronounced natural oscillation
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # Quadratic air drag model (proportional to velocity squared)
        # More physically accurate than linear drag for higher velocities
        air_resistance = self.air_drag * theta_dot**2 * np.sign(-theta_dot)

        # Calculate denominator term (common in both equations)
        d = self.I * (self.M + self.m) + self.M * self.m * self.L**2 * sin_theta**2

        # Calculate accelerations with air drag effects
        x_ddot = (u - self.b * x_dot - self.m * self.L * theta_dot**2 * sin_theta - 
              self.m * self.L * cos_theta * (self.m * self.g * self.L * sin_theta - air_resistance) / d) / \
             (self.M + self.m - self.m * self.L * cos_theta**2 * self.m / d)

        theta_ddot = (self.m * self.g * self.L * sin_theta - air_resistance - 
                  self.m * self.L * cos_theta * x_ddot) / d

        return [x_dot, x_ddot, theta_dot, theta_ddot]
    
    def linearize(self):
        """
        Linearize the system around the equilibrium point (upright position).
        
        Returns:
            A: System matrix
            B: Input matrix
        """
        # For small angles: sin(θ) ≈ θ, cos(θ) ≈ 1
        # The linearized system matrices
        A = np.array([
            [0, 1, 0, 0],
            [0, -self.b/(self.M+self.m), self.m*self.g*self.L/(self.M+self.m), 0],
            [0, 0, 0, 1],
            [0, -self.b*self.L/((self.M+self.m)*self.L**2 + self.I), 
             self.g*(self.M+self.m)*self.L/((self.M+self.m)*self.L**2 + self.I), 0]
        ])
        
        B = np.array([
            [0],
            [1/(self.M+self.m)],
            [0],
            [self.L/((self.M+self.m)*self.L**2 + self.I)]
        ])
        
        return A, B


def kalman_filter(z, x_prev, P_prev, F, H, Q, R):
    """
    Kalman filter implementation for state estimation with noisy measurements.
    
    Args:
        z: Measurement vector
        x_prev: Previous state estimate
        P_prev: Previous error covariance
        F: State transition matrix
        H: Measurement matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        
    Returns:
        x: Updated state estimate
        P: Updated error covariance
    """
    # Predict
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.T + Q
    
    # Update
    y = z - H @ x_pred  # Measurement residual
    S = H @ P_pred @ H.T + R  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    
    x = x_pred + K @ y  # Updated state estimate
    P = (np.eye(len(x_pred)) - K @ H) @ P_pred  # Updated error covariance
    
    return x, P
def ziegler_nichols_tune(model, method="classic"):
    """
    Auto-tune PID controller using Ziegler-Nichols method.
    
    Args:
        model: The system model
        method: Tuning method ("classic" or "refined")
        
    Returns:
        Kp, Ki, Kd: Tuned PID gains
    """
    # Get linearized model
    A, B = model.linearize()
    
    # Estimate ultimate gain and period
    # This is a simplified approach - in practice, you'd use the oscillation method
    # Here we use pole placement to find the stability limit
    
    # Find ultimate gain (Ku) - gain that causes sustained oscillation
    Ku = 0
    for gain in np.arange(1.0, 100.0, 0.5):
        # Create closed-loop system with proportional-only control
        A_cl = A - gain * B @ np.array([[0, 0, 1, 0]])
        eigenvalues = np.linalg.eigvals(A_cl)
        # Check if any eigenvalues are on imaginary axis (sustained oscillation)
        if any(abs(np.real(ev)) < 0.01 and abs(np.imag(ev)) > 0.01 for ev in eigenvalues):
            Ku = gain
            # Estimate oscillation period from imaginary part
            for ev in eigenvalues:
                if abs(np.real(ev)) < 0.01 and np.imag(ev) > 0.01:
                    Tu = 2 * np.pi / np.imag(ev)
                    break
            break
    
    # If we couldn't find Ku with this method, use a fallback value
    if Ku == 0:
        print("Could not determine ultimate gain, using fallback values")
        Ku = 30.0
        Tu = 0.5
    
    print(f"Ziegler-Nichols parameters: Ku={Ku:.2f}, Tu={Tu:.4f}s")
    
    # Apply Ziegler-Nichols formulas
    if method == "classic":
        # Classic Ziegler-Nichols
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu
    elif method == "refined":
        # Some refined PID tuning rules (Tyreus-Luyben)
        Kp = 0.45 * Ku
        Ki = 0.54 * Ku / Tu
        Kd = 0.075 * Ku * Tu
    else:
        # Default to reasonable values
        Kp = 15.0
        Ki = 0.5
        Kd = 5.0
    
    return Kp, Ki, Kd

def main():
    # Connect to PyBullet and set up environment
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Load the cart-pole URDF
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=False)

    # Set up PyBullet parameters for better interaction
    p.setPhysicsEngineParameter(enableFileCaching=0,
                                numSolverIterations=50,
                                numSubSteps=4)
    
    # Simulation parameters
    dt = 1.0 / 240.0
    max_sim_time = 120.0  # Extended simulation time
    
    # Disable default motor control and get joint information
    num_joints = p.getNumJoints(cart_id)
    for j in range(num_joints):
        p.setJointMotorControl2(cart_id, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # Identify joints: pole and wheels
    wheel_names = ["tl_base_joint", "bl_base_joint", "tr_base_joint", "br_base_joint"]
    wheel_indices = []
    pole_joint_index = None
    
    for j in range(num_joints):
        info = p.getJointInfo(cart_id, j)
        joint_name = info[1].decode("utf-8")
        if joint_name in wheel_names:
            wheel_indices.append(j)
        elif joint_name == "pole_base_joint":
            pole_joint_index = j

    if pole_joint_index is None:
        print("Error: 'pole_base_joint' not found.")
        p.disconnect()
        return
    if len(wheel_indices) != 4:
        print("Error: Could not find all 4 wheel joints.")
        p.disconnect()
        return

    print("Wheel joints:", wheel_indices)
    print("Pole joint index:", pole_joint_index)

    # Initialize the pendulum model
    pendulum_model = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5)
    A, B = pendulum_model.linearize()

    # Auto-tune PID controller
    Kp, Ki, Kd = ziegler_nichols_tune(pendulum_model, "refined")
    print(f"Auto-tuned PID parameters: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")

    # PID controller with auto-tuned gains
    pid_controller = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0.0, 
                               output_limits=(-20.0, 20.0), dead_zone=0.01)

    # Pole Placement controller setup
    desired_poles = [-1.0, -2.0, -3.0, -4.0]  # Stable poles
    pole_placement_controller = PolePlacementController(A, B, desired_poles, output_limits=(-20.0, 20.0))

    # Nonlinear controller setup
    nonlinear_controller = NonlinearController(output_limits=(-20.0, 20.0))
    
    # Initialize with a larger initial angle to see natural oscillation clearly
    initial_angle = 0.2  # increased from 0.1
    p.resetJointState(cart_id, pole_joint_index, initial_angle)

    # Adjust dynamics for more realistic behavior
    for i in range(-1, num_joints):
        p.changeDynamics(cart_id, i, 
                    lateralFriction=0.8,  # Reduced from 1.0
                    rollingFriction=0.005,  # Reduced from 0.01
                    spinningFriction=0.005,  # Reduced from 0.01
                    restitution=0.2)  # Increased from 0.1

    # Setup debug parameters
    control_type_id = p.addUserDebugParameter("Controller (0:Off, 1:PID, 2:Pole Place, 3:Nonlinear)", 0, 3, 0)
    noise_amplitude_id = p.addUserDebugParameter("Sensor Noise (0-1)", 0, 1, 0)
    filter_strength_id = p.addUserDebugParameter("Filter Strength (0-1)", 0, 1, 0.5)
    
    # Add controller tuning parameters
    pid_p_id = p.addUserDebugParameter("PID P Gain", 0, 50, 15)
    pid_i_id = p.addUserDebugParameter("PID I Gain", 0, 5, 0.5)
    pid_d_id = p.addUserDebugParameter("PID D Gain", 0, 20, 5)
    
    # Disturbance parameter
    disturbance_id = p.addUserDebugParameter("Apply Disturbance", -20, 20, 0)
    
    # Phase control - natural oscillation first, then control
    natural_oscillation_time_id = p.addUserDebugParameter("Natural Oscillation Time", 0, 10, 5)
    
    # Setup for real-time plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    cart_positions = []
    pendulum_angles = []
    control_forces = []
    timestamps = []
    
    # State estimation setup for Kalman Filter
    # Initialize state and covariance
    x_est = np.array([[0], [0], [initial_angle], [0]])
    P_est = np.eye(4) * 0.1
    
    # Define noise covariances for Kalman filter
    Q_kalman = np.eye(4) * 0.01  # Process noise
    R_kalman = np.eye(4) * 0.1   # Measurement noise
    
    # Lines for plotting
    time_data = np.linspace(0, 10, 100)  # 10 seconds of data initially
    line_pos, = ax1.plot(time_data, np.zeros_like(time_data))
    line_angle, = ax2.plot(time_data, np.zeros_like(time_data))
    line_force, = ax3.plot(time_data, np.zeros_like(time_data))
    
    # Set up the plots
    ax1.set_ylabel('Cart Position (m)')
    ax1.set_ylim(-2, 2)
    ax1.grid(True)
    
    ax2.set_ylabel('Pendulum Angle (rad)')
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.set_ylim(-20, 20)
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Text for displaying current controller and parameters
    ctrl_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes)
    phase_text = ax1.text(0.02, 0.85, "Phase: Natural oscillation", transform=ax1.transAxes)
    
    sim_time = 0
    prev_time = time.time()
    
    # Create a debug line to visualize the force applied
    force_visual_id = p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0, 0], lineWidth=2)
    
    # Phase state
    natural_oscillation_phase = True
    phase_text_id = p.addUserDebugText("Phase: Natural Oscillation", [0, 0, 0.5], [1, 1, 1], textSize=1.5)
    
    # Main simulation loop
    while sim_time < max_sim_time:
        # Get updated parameters from UI
        control_type = int(p.readUserDebugParameter(control_type_id))
        noise_amplitude = p.readUserDebugParameter(noise_amplitude_id)
        filter_strength = p.readUserDebugParameter(filter_strength_id)
        natural_oscillation_time = p.readUserDebugParameter(natural_oscillation_time_id)
        
        # Check if we should switch from natural oscillation to control
        if natural_oscillation_phase and sim_time >= natural_oscillation_time:
            natural_oscillation_phase = False
            p.addUserDebugText("Phase: Active Control", [0, 0, 0.5], [1, 1, 1], 
                              textSize=1.5, replaceItemUniqueId=phase_text_id)
            phase_text.set_text("Phase: Active Control")
        
        # Update PID parameters if changed
        pid_controller.Kp = p.readUserDebugParameter(pid_p_id)
        pid_controller.Ki = p.readUserDebugParameter(pid_i_id)
        pid_controller.Kd = p.readUserDebugParameter(pid_d_id)
        
        # Get the disturbance force
        disturbance = p.readUserDebugParameter(disturbance_id)
        
        # Get joint states
        cart_pos, cart_vel = p.getBasePositionAndOrientation(cart_id)[0][0], p.getBaseVelocity(cart_id)[0][0]
        pole_state = p.getJointState(cart_id, pole_joint_index)
        pole_angle, pole_vel = pole_state[0], pole_state[1]
        
        # Add simulated sensor noise
        if noise_amplitude > 0:
            noisy_pole_angle = pole_angle + np.random.normal(0, noise_amplitude * 0.05)
            noisy_cart_pos = cart_pos + np.random.normal(0, noise_amplitude * 0.02)
        else:
            noisy_pole_angle = pole_angle
            noisy_cart_pos = cart_pos
        
        # State vector for control
        state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
        noisy_state = np.array([noisy_cart_pos, cart_vel, noisy_pole_angle, pole_vel])
        
        # Apply Kalman filtering if filter strength > 0
        if filter_strength > 0:
            # Update measurement noise based on UI setting
            R_kalman = np.eye(4) * (0.1 + noise_amplitude * 0.2)
            
            # State transition matrix F (approximately A*dt + I)
            F = np.eye(4) + A * dt
            
            # Measurement matrix (we observe all states)
            H = np.eye(4)
            
            # Filter the noisy state
            x_est, P_est = kalman_filter(
                noisy_state.reshape(-1, 1), 
                x_est, 
                P_est, 
                F, 
                H, 
                Q_kalman, 
                R_kalman
            )
            
            # Extract the filtered state
            filtered_state = np.array([x_est[0, 0], x_est[1, 0], x_est[2, 0], x_est[3, 0]])
            
            # Blend between noisy and filtered state based on filter strength
            control_state = filtered_state * filter_strength + noisy_state * (1 - filter_strength)
        else:
            control_state = noisy_state
        
        # Compute control force based on selected controller
        force = 0

        # Only apply control if not in natural oscillation phase
        if not natural_oscillation_phase:
            if control_type == 1:  # PID
                output, error = pid_controller.compute(control_state[2], dt)  # Control based on angle
                force = output
            elif control_type == 2:  # Pole Placement
                force = pole_placement_controller.compute(control_state)
            elif control_type == 3:  # Nonlinear
                force = nonlinear_controller.compute(control_state)
        
        # Apply disturbance regardless of phase
        force += disturbance
        
        # Apply the control force to all wheels
        for wheel_idx in wheel_indices:
            p.applyExternalForce(
                cart_id, 
                wheel_idx, 
                [force, 0, 0], 
                [0, 0, 0], 
                p.LINK_FRAME
            )
        
        # Update force visualization line
        cart_position, _ = p.getBasePositionAndOrientation(cart_id)
        line_start = [cart_position[0], cart_position[1], 0.05]
        line_end = [cart_position[0] + 0.05 * force, cart_position[1], 0.05]
        p.addUserDebugLine(line_start, line_end, [1, 0, 0], lineWidth=2, replaceItemUniqueId=force_visual_id)
        
        # Store data for plotting
        cart_positions.append(cart_pos)
        pendulum_angles.append(pole_angle)
        control_forces.append(force)
        timestamps.append(sim_time)
        
        # Only keep the last 1000 points for efficiency
        if len(timestamps) > 1000:
            cart_positions.pop(0)
            pendulum_angles.pop(0)
            control_forces.pop(0)
            timestamps.pop(0)
        
        # Update plots every 100 timesteps to avoid slowing down simulation
        if len(timestamps) % 100 == 0:
            plot_time_window = 10  # seconds of data to display
            
            if len(timestamps) > 1:
                # Focus on the most recent plot_time_window seconds of data
                start_idx = 0
                if timestamps[-1] > plot_time_window:
                    for i, t in enumerate(timestamps):
                        if timestamps[-1] - t <= plot_time_window:
                            start_idx = i
                            break
                
                x_data = timestamps[start_idx:]
                x_min, x_max = x_data[0], x_data[-1]
                
                # Update lines with latest data
                line_pos.set_data(x_data, cart_positions[start_idx:])
                line_angle.set_data(x_data, pendulum_angles[start_idx:])
                line_force.set_data(x_data, control_forces[start_idx:])
                
                # Update x-axis limits to maintain a moving window
                ax1.set_xlim(x_min, x_max)
                ax2.set_xlim(x_min, x_max)
                ax3.set_xlim(x_min, x_max)
                
                # Update controller info text
                controller_names = {0: "Off", 1: "PID", 2: "LQR", 3: "MPC", 4: "Pole Placement", 5: "Nonlinear"}
                ctrl_info = f"Controller: {controller_names.get(control_type, 'Off')}"
                if noise_amplitude > 0:
                    ctrl_info += f", Noise: {noise_amplitude:.2f}"
                if filter_strength > 0:
                    ctrl_info += f", Filter: {filter_strength:.2f}"
                ctrl_text.set_text(ctrl_info)
                
                # Redraw the figure
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
        
        # Step the simulation
        p.stepSimulation()
        
        # Calculate real time step and update sim_time
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        sim_time += frame_time
        
        # Maintain a reasonable simulation speed
        time_to_sleep = max(0, dt - frame_time)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        
        # Check if pendulum has fallen beyond recovery or cart is too far
        if abs(pole_angle) > 0.8 or abs(cart_pos) > 3.0:
            print("Simulation reset: pendulum fell or cart went too far")
            p.resetJointState(cart_id, pole_joint_index, 0.2)
            for wheel_idx in wheel_indices:
                p.resetJointState(cart_id, wheel_idx, 0)
            p.resetBasePositionAndOrientation(cart_id, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            
            # Reset controllers
            pid_controller.reset()
            
            # Reset state
            natural_oscillation_phase = True
            p.addUserDebugText("Phase: Natural Oscillation", [0, 0, 0.5], [1, 1, 1], 
                               textSize=1.5, replaceItemUniqueId=phase_text_id)
            phase_text.set_text("Phase: Natural Oscillation")
            
            # Clear all plots for a fresh start
            cart_positions.clear()
            pendulum_angles.clear()
            control_forces.clear()
            timestamps.clear()
            
            # Reset simulation time to make plotting restart
            sim_time = 0
    
    # Clean up resources
    p.disconnect()
    plt.ioff()
    plt.close('all')

# Add this function after compare_controllers() around line ~1500
def analyze_air_drag_effects():
    """Analyze and visualize the effect of air drag on the pendulum system."""
    print("Analyzing air drag effects...")
    
    # Set up simulation parameters
    dt = 0.01
    sim_time = 10.0  # seconds
    t = np.arange(0, sim_time, dt)
    
    # Initial conditions: small displacement from upright position
    initial_state = [0, 0, 0.2, 0]  # [x, x_dot, theta, theta_dot]
    
    # Create pendulum models with and without air drag
    pendulum_no_drag = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5, air_drag=0.0)
    pendulum_with_drag = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5, air_drag=0.05)
    
    # Solve the ODE for both models
    sol_no_drag = solve_ivp(
        lambda t, y: pendulum_no_drag.dynamics(t, y, 0.0),
        [0, sim_time],
        initial_state,
        t_eval=t
    )
    
    sol_with_drag = solve_ivp(
        lambda t, y: pendulum_with_drag.dynamics(t, y, 0.0),
        [0, sim_time],
        initial_state,
        t_eval=t
    )
    
    # Extract pendulum angles
    theta_no_drag = sol_no_drag.y[2]
    theta_with_drag = sol_with_drag.y[2]
    
    # Calculate energy for both cases
    def calc_energy(model, state_history):
        energy = []
        for i in range(len(state_history[0])):
            x, x_dot, theta, theta_dot = state_history[:, i]
            
            # Kinetic energy of cart and pendulum
            ke_cart = 0.5 * model.M * x_dot**2
            ke_pendulum = 0.5 * model.m * ((x_dot + model.L * theta_dot * np.cos(theta))**2 + 
                                          (model.L * theta_dot * np.sin(theta))**2)
            
            # Potential energy of pendulum (zero at the lowest point)
            pe_pendulum = model.m * model.g * model.L * (1 - np.cos(theta))
            
            total_energy = ke_cart + ke_pendulum + pe_pendulum
            energy.append(total_energy)
        return np.array(energy)
    
    energy_no_drag = calc_energy(pendulum_no_drag, sol_no_drag.y)
    energy_with_drag = calc_energy(pendulum_with_drag, sol_with_drag.y)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot pendulum angles
    ax1.plot(t, theta_no_drag, 'b-', label='Without Air Drag')
    ax1.plot(t, theta_with_drag, 'r-', label='With Air Drag')
    ax1.set_ylabel('Pendulum Angle (rad)')
    ax1.set_title('Effect of Air Drag on Pendulum Oscillation')
    ax1.grid(True)
    ax1.legend()
    
    # Plot system energy
    ax2.plot(t, energy_no_drag, 'b-', label='Without Air Drag')
    ax2.plot(t, energy_with_drag, 'r-', label='With Air Drag')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Total Energy (J)')
    ax2.set_title('System Energy Conservation/Dissipation')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("air_drag_effects.png", dpi=300)
    plt.show()
    
    # Calculate and print energy dissipation rate
    energy_loss_rate_drag = (energy_with_drag[0] - energy_with_drag[-1]) / sim_time
    print(f"Energy dissipation rate with air drag: {energy_loss_rate_drag:.4f} J/s")
    
    # Log the frequency information
    # Natural frequency calculation
    if len(t) > 2:
        # Find peaks to estimate frequency
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(theta_no_drag)
        if len(peaks) >= 2:
            avg_period = np.mean(np.diff(t[peaks]))
            natural_freq = 1/avg_period if avg_period > 0 else 0
            print(f"Estimated natural frequency: {natural_freq:.4f} Hz")
            
            # Calculate damping ratio from amplitude decay
            if len(peaks) >= 3:
                amplitudes = theta_no_drag[peaks]
                decay_ratio = amplitudes[0] / amplitudes[-1]
                damping_ratio = np.log(decay_ratio) / (2 * np.pi * len(peaks))
                print(f"Estimated damping ratio without air drag: {damping_ratio:.4f}")
                
            # With air drag
            peaks_drag, _ = find_peaks(theta_with_drag)
            if len(peaks_drag) >= 3:
                amplitudes_drag = theta_with_drag[peaks_drag]
                decay_ratio_drag = amplitudes_drag[0] / amplitudes_drag[-1]
                damping_ratio_drag = np.log(decay_ratio_drag) / (2 * np.pi * len(peaks_drag))
                print(f"Estimated damping ratio with air drag: {damping_ratio_drag:.4f}")

def update_phase_visualization(phase, phase_text_id):
    """Update the phase visualization text."""
    phase_text = f"Phase: {phase.capitalize()}"
    p.addUserDebugText(phase_text, [0, 0, 0.5], [1, 1, 1], textSize=1.5, replaceItemUniqueId=phase_text_id)

def record_experiment(controller_type, use_noise=False, disturbance_magnitude=10.0, duration=20.0):
    """
    Run a controlled experiment and record data for analysis.
    
    Args:
        controller_type: 1=PID, 2=LQR, 3=MPC, 4=PolePlace, 5=Nonlinear
        use_noise: Whether to add sensor noise
        disturbance_magnitude: Magnitude of disturbance force
        duration: Duration of recording in seconds
    
    Returns:
        Dictionary with recorded data
    """
    # Setup similar to main()
    p.connect(p.DIRECT)  # Headless mode for faster recording
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load cart and pendulum
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.setAdditionalSearchPath(current_dir)
    cart_id = p.loadURDF("flagpole.urdf", [0, 0, 0.1], useFixedBase=False)
    
    # Simulation parameters
    dt = 1.0 / 240.0
    
    # Find joints
    wheel_indices = []
    pole_joint_index = None
    for j in range(p.getNumJoints(cart_id)):
        info = p.getJointInfo(cart_id, j)
        joint_name = info[1].decode("utf-8")
        if "base_joint" in joint_name and joint_name != "pole_base_joint":
            wheel_indices.append(j)
        elif joint_name == "pole_base_joint":
            pole_joint_index = j

    # Initialize pendulum and controllers
    pendulum_model = InvertedPendulumModel(M=10.0, m=1.0, L=0.5, b=0.5)
    A, B = pendulum_model.linearize()
    
    # Create controllers
    pid_controller = PIDController(Kp=15.0, Ki=0.5, Kd=5.0, setpoint=0.0, output_limits=(-20.0, 20.0))
    
    # Pole Placement controller
    desired_poles = [-1.0, -2.0, -3.0, -4.0]  # Stable poles
    pole_placement_controller = PolePlacementController(A, B, desired_poles, output_limits=(-20.0, 20.0))
    
    # Nonlinear controller
    nonlinear_controller = NonlinearController(output_limits=(-20.0, 20.0))
    
    # Initialize with a larger initial angle to observe natural oscillation
    initial_angle = 0.2
    p.resetJointState(cart_id, pole_joint_index, initial_angle)
    
    # Data storage
    data = {
        'time': [],
        'cart_position': [],
        'pendulum_angle': [],
        'control_force': [],
        'natural_oscillation': []  # Flag for natural oscillation phase
    }
    
    # Simulation loop
    sim_time = 0
    natural_oscillation_phase = True
    phase_switch_time = 2.0  # Switch to control after 2 seconds of natural oscillation
    
    while sim_time < duration:
        # Switch from natural oscillation to control
        if natural_oscillation_phase and sim_time >= phase_switch_time:
            natural_oscillation_phase = False
        
        # Get states
        cart_pos, cart_vel = p.getBasePositionAndOrientation(cart_id)[0][0], p.getBaseVelocity(cart_id)[0][0]
        pole_state = p.getJointState(cart_id, pole_joint_index)
        pole_angle, pole_vel = pole_state[0], pole_state[1]
        
        # Add simulated sensor noise if enabled
        if use_noise:
            noise_amplitude = 0.05  # Moderate noise
            noisy_cart_pos = cart_pos + np.random.normal(0, noise_amplitude * 0.02)
            noisy_pole_angle = pole_angle + np.random.normal(0, noise_amplitude * 0.05)
        else:
            noisy_cart_pos = cart_pos
            noisy_pole_angle = pole_angle
        
        state = np.array([cart_pos, cart_vel, pole_angle, pole_vel])
        noisy_state = np.array([noisy_cart_pos, cart_vel, noisy_pole_angle, pole_vel])
        
        # Compute control force
        force = 0
        if not natural_oscillation_phase:  # Only apply control after initial natural oscillation
            if controller_type == 1:  # PID
                output, _ = pid_controller.compute(noisy_pole_angle, dt)
                force = output
            elif controller_type == 2:  # Pole Placement
                force = pole_placement_controller.compute(noisy_state)
            elif controller_type == 3:  # Nonlinear
                force = nonlinear_controller.compute(noisy_state)
        
        # Apply disturbance at specific time
        if 5.0 <= sim_time <= 5.2:  # Short, strong disturbance
            force += disturbance_magnitude
        
        # Apply the control force to all wheels
        for wheel_idx in wheel_indices:
            p.applyExternalForce(cart_id, wheel_idx, [force, 0, 0], [0, 0, 0], p.LINK_FRAME)
        
        # Record data
        data['time'].append(sim_time)
        data['cart_position'].append(cart_pos)
        data['pendulum_angle'].append(pole_angle)
        data['control_force'].append(force)
        data['natural_oscillation'].append(natural_oscillation_phase)
        
        # Step simulation
        p.stepSimulation()
        sim_time += dt
    
    p.disconnect()
    return data


def compare_controllers(use_noise=False, disturbance_magnitude=10.0):
    """
    Compare and visualize the performance of different controllers.
    
    Args:
        use_noise: Whether to add sensor noise
        disturbance_magnitude: Magnitude of disturbance force
    """
    # Record data for each controller
    pid_data = record_experiment(1, use_noise, disturbance_magnitude)
    pole_place_data = record_experiment(2, use_noise, disturbance_magnitude)
    nonlinear_data = record_experiment(3, use_noise, disturbance_magnitude)
    
    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Cart position
    ax1.plot(pid_data['time'], pid_data['cart_position'], 'r-', label='PID')
    ax1.plot(lqr_data['time'], lqr_data['cart_position'], 'g-', label='LQR')
    ax1.plot(mpc_data['time'], mpc_data['cart_position'], 'b-', label='MPC')
    ax1.plot(pole_place_data['time'], pole_place_data['cart_position'], 'm-', label='Pole Placement')
    ax1.plot(nonlinear_data['time'], nonlinear_data['cart_position'], 'c-', label='Nonlinear')
    ax1.set_ylabel('Cart Position (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Pendulum angle
    ax2.plot(pid_data['time'], pid_data['pendulum_angle'], 'r-', label='PID')
    ax2.plot(lqr_data['time'], lqr_data['pendulum_angle'], 'g-', label='LQR')
    ax2.plot(mpc_data['time'], mpc_data['pendulum_angle'], 'b-', label='MPC')
    ax2.plot(pole_place_data['time'], pole_place_data['pendulum_angle'], 'm-', label='Pole Placement')
    ax2.plot(nonlinear_data['time'], nonlinear_data['pendulum_angle'], 'c-', label='Nonlinear')
    ax2.set_ylabel('Pendulum Angle (rad)')
    ax2.grid(True)
    
    # Control force
    ax3.plot(pid_data['time'], pid_data['control_force'], 'r-', label='PID')
    ax3.plot(lqr_data['time'], lqr_data['control_force'], 'g-', label='LQR')
    ax3.plot(mpc_data['time'], mpc_data['control_force'], 'b-', label='MPC')
    ax3.plot(pole_place_data['time'], pole_place_data['control_force'], 'm-', label='Pole Placement')
    ax3.plot(nonlinear_data['time'], nonlinear_data['control_force'], 'c-', label='Nonlinear')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Force (N)')
    ax3.grid(True)
    
    # Add vertical lines to show when natural oscillation ends and disturbance occurs
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=2.0, color='k', linestyle='--', alpha=0.5, label='Control Start')
        ax.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label='Disturbance')
    
    # Add legend to top plot only
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='upper right')
    
    # Calculate performance metrics
    def calc_metrics(data):
        # Response time (time to stabilize within 0.05 radians)
        angle_data = np.array(data['pendulum_angle'])
        time_data = np.array(data['time'])
        
        # Find where system stabilizes after disturbance at t=5s
        if len(time_data) > 0:
            disturbance_idx = np.argmin(np.abs(time_data - 5.0))
            for i in range(disturbance_idx, len(angle_data)):
                if i+100 < len(angle_data) and abs(angle_data[i]) < 0.05 and all(abs(angle_data[i:i+100]) < 0.05):
                    response_time = time_data[i] - 5.0
                    break
            else:
                response_time = float('inf')
                
            # Calculate RMS error after initial settling (2-5s)
            settle_start = np.argmin(np.abs(time_data - 2.0))
            settle_end = np.argmin(np.abs(time_data - 5.0))
            if settle_start < settle_end:
                rms_error = np.sqrt(np.mean(np.square(angle_data[settle_start:settle_end])))
            else:
                rms_error = float('inf')
            
            # Calculate maximum overshoot after disturbance
            if disturbance_idx < len(angle_data):
                max_overshoot = np.max(np.abs(angle_data[disturbance_idx:]))
            else:
                max_overshoot = float('inf')
            
            # Calculate control effort (sum of absolute forces)
            control_effort = np.sum(np.abs(data['control_force']))
            
            return {
                'response_time': response_time, 
                'rms_error': rms_error,
                'max_overshoot': max_overshoot,
                'control_effort': control_effort
            }
        return {
            'response_time': float('inf'), 
            'rms_error': float('inf'),
            'max_overshoot': float('inf'),
            'control_effort': float('inf')
        }
    
    pid_metrics = calc_metrics(pid_data)
    lqr_metrics = calc_metrics(lqr_data)
    mpc_metrics = calc_metrics(mpc_data)
    pole_place_metrics = calc_metrics(pole_place_data)
    nonlinear_metrics = calc_metrics(nonlinear_data)
    
    # Create summary table
    metrics_table = {
        'Controller': ['PID', 'LQR', 'MPC', 'Pole Placement', 'Nonlinear'],
        'Response Time (s)': [
            pid_metrics['response_time'], 
            lqr_metrics['response_time'],
            mpc_metrics['response_time'],
            pole_place_metrics['response_time'],
            nonlinear_metrics['response_time']
        ],
        'RMS Error': [
            pid_metrics['rms_error'], 
            lqr_metrics['rms_error'],
            mpc_metrics['rms_error'],
            pole_place_metrics['rms_error'],
            nonlinear_metrics['rms_error']
        ],
        'Max Overshoot': [
            pid_metrics['max_overshoot'], 
            lqr_metrics['max_overshoot'],
            mpc_metrics['max_overshoot'],
            pole_place_metrics['max_overshoot'],
            nonlinear_metrics['max_overshoot']
        ],
        'Control Effort': [
            pid_metrics['control_effort'], 
            lqr_metrics['control_effort'],
            mpc_metrics['control_effort'],
            pole_place_metrics['control_effort'],
            nonlinear_metrics['control_effort']
        ]
    }
    
    # Add metrics to plot title
    noise_text = " with Noise" if use_noise else ""
    plt.suptitle(f"Controller Comparison{noise_text} - Response to Disturbance", fontsize=14)
    
    # Add a table below the plots
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot
    filename = f"controller_comparison{'_with_noise' if use_noise else ''}.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    # Print metrics table
    print("\nController Performance Metrics:")
    print(f"{'Controller':<15} {'Response Time (s)':<20} {'RMS Error':<15} {'Max Overshoot':<15} {'Control Effort':<15}")
    print("-" * 80)
    for i, controller in enumerate(metrics_table['Controller']):
        print(f"{controller:<15} {metrics_table['Response Time (s)'][i]:<20.3f} {metrics_table['RMS Error'][i]:<15.5f} {metrics_table['Max Overshoot'][i]:<15.5f} {metrics_table['Control Effort'][i]:<15.0f}")

    # Return metrics for further analysis
    return {
        'PID': pid_metrics,
        'LQR': lqr_metrics,
        'MPC': mpc_metrics,
        'Pole Placement': pole_place_metrics,
        'Nonlinear': nonlinear_metrics
    }

if __name__ == "__main__":
    main()
    # Uncomment to run controller comparison
    compare_controllers()