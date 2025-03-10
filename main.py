"""Inverted Pendulum Simulation."""

from pendulum_physics import InvertedPendulum
from visualization import ContinuousSimulation


def main():
    """Main entry point for the Inverted Pendulum Simulation."""
    print("\n=== Inverted Pendulum Simulation ===")
    sim = ContinuousSimulation()
    sim.main_loop()


if __name__ == "__main__":
    main()
