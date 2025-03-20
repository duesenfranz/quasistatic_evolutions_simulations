

from quasistatic_evol.rod_simulator import load_simulations, SimulationStepResult, SystemState


def print_breaking_times():
    """
    Print the breaking times of the different models.
    """
    for name, simulation in load_simulations().items():
        breaking_step = max(simulation, key=lambda result: result.energy_before_system_transition - result.energy)
        print(f"{name} breaks at {breaking_step.time}")

if __name__ == "__main__":
    print_breaking_times()