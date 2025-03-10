import numpy as np
import matplotlib.pyplot as plt


def create_scenario_plot(results, noise_on=False, filter_on=False):
    """Create and save a scenario plot based on simulation results."""
    plt.figure(figsize=(16, 12))

    # Subplot 1: Angle Response
    plt.subplot(2, 2, 1)
    for controller, data in results.items():
        plt.plot(data['time'], data['angle'], label=controller)
    plt.axhline(y=2, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle Response to Disturbance')
    plt.legend()

    # Subplot 2: Control Effort
    plt.subplot(2, 2, 2)
    for controller, data in results.items():
        plt.plot(data['time'], data['control'], label=controller)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Force (N)')
    plt.title('Control Effort')
    plt.legend()

    # Subplot 3: Cart Position
    plt.subplot(2, 2, 3)
    for controller, data in results.items():
        plt.plot(data['time'], data['position'], label=controller)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Cart Position (m)')
    plt.title('Cart Position')
    plt.legend()

    # Subplot 4: Metrics Table
    plt.subplot(2, 2, 4)
    plt.axis('off')

    metrics = [
        [
            'Controller',
            'Response Time (s)',
            'Settling Time (s)',
            'Max Deviation (°)',
            'Control Effort (N·s)'
        ]
    ]

    for controller, data in results.items():
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

    plt.title(
        f"Performance Metrics (Noise: {'ON' if noise_on else 'OFF'}, "
        f"Filter: {'ON' if filter_on else 'OFF'})"
    )

    plt.suptitle(
        f"Controller Comparison - Noise {'ON' if noise_on else 'OFF'}, "
        f"Filter {'ON' if filter_on else 'OFF'}",
        fontsize=16
    )

    plt.tight_layout()

    scenario_name = (
        f"scenario_noise{'on' if noise_on else 'off'}_"
        f"filter{'on' if filter_on else 'off'}.png"
    )
    plt.savefig(scenario_name)

    return plt.gcf()
