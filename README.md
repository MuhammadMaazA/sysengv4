# Inverted Pendulum Simulation

A physics-based simulation of an inverted pendulum with different control methods.

## What's this about?

This is a simulation of an inverted pendulum on a cart - basically, it shows how different control strategies can balance a stick that wants to fall over. You can play with PID, pole placement, and nonlinear controllers and see how they handle disturbances and sensor noise.

## Getting Started

### Prerequisites

Make sure you have Python 3.6+ and the following packages installed:

- `numpy`
- `matplotlib`
- `scipy`

To install them, run:

```bash
pip install numpy matplotlib scipy
```

### Running the Simulation

Save all the files in the same folder:

- `main.py`
- `pendulum_physics.py`
- `controllers.py`
- `visualization.py`
- `analysis_utils.py`

Run the main script:

```bash
python main.py
```

## Playing with the Simulation

### Controls

- **Impulse slider:** Apply a force to the cart by dragging and releasing.
- **PID Gains:** Adjust Kp, Ki, and Kd values to tune the PID controller.
- **Controller selection:** Choose between PID, Pole placement, and Nonlinear controllers.
- **Noise/Filter:** Toggle sensor noise and filtering on/off.
- **Reset:** Reset the simulation to the starting position.
- **Run All Tests:** Generate comparison plots for all three controllers under different conditions.

### The Interface

- **Top display:** Shows the cart and pendulum, along with the current angle.
- **Middle left graph:** Displays system response (angle, position, control force).
- **Middle right graph:** Displays any disturbance forces.
- **Control panel (bottom):** Allows parameter adjustments.

## What to Try

- Apply an impulse disturbance and observe how each controller recovers.
- Turn on noise and analyze its effect on control.
- Enable filtering to see how it helps reduce noise impact.
- Click **"Run All Tests"** to generate comparison plots for all controllers.

## What the Tests Show

When you click **"Run All Tests"**, the program runs simulations for these scenarios:

1. **Noise ON, Filter ON**
2. **Noise OFF, Filter OFF**
3. **Noise ON, Filter OFF**

For each scenario, the following details are generated:

- **Angle responses to disturbances**
- **Control effort used by each controller**
- **Cart position over time**
- **Performance metrics table**

Results are saved as PNG files for use in reports or presentations.

## Understanding the Results

- **Response Time:** How quickly the controller recovers (smaller is better).
- **Settling Time:** Time taken for oscillations to stop (smaller is better).
- **Max Deviation:** Maximum angle reached during disturbance.
- **Control Effort:** Total force used (smaller means more efficient).

Each controller has different strengths:
- **PID:** Simple but effective.
- **Pole Placement:** Offers good settling times.
- **Nonlinear Controller:** Features interesting energy-based behavior.