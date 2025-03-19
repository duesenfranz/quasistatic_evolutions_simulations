import json
from pathlib import Path
import pickle
import torch
from torch import Tensor
from matplotlib import colorbar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import axes
from matplotlib import figure
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import os
from typing import Literal, Tuple


@dataclass
class SimulationConfig:
    H: float = 1.0
    W: float = 10.0
    N: int = 30
    C_surface_average: float = 0.03
    C_surface_std: float = 0.002
    L: float = W / N / 1.3
    k_stretch: float = 0.2 / L
    c_angle: float = 0.0
    learning_rate: float = 0.1
    decay_factor: float = 1
    min_learning_rate: float = 0.001
    tolerance: float = 5e-4
    max_iterations: int = 100000
    d: float = 5
    time_steps: int = 20
    plot_interval: int = N // 10
    error_norm_p: int = 2

@dataclass
class SystemState:
    free_nodes: Tensor
    breakages: Tensor

@dataclass
class SimulationStepResult:
    energy_before_system_transition: float
    energy: float
    energy_gradient: float
    SystemState: SystemState
    time: float
    iterations: int




class Simulator:
    def __init__(self, config: SimulationConfig, seed: int = 0):
        """
        Initialize the Simulator.

        :param config: Configuration for the simulation
        :param seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.fig: figure.Figure | None = None
        self.ax: axes.Axes | None = None
        self.surface_energy_constants = self.initialize_surface_energy()
        self.free_rod_nodes_indices = [False] + [True] * (self.config.N - 2) + [False]
        self.controlled_nodes_indices = [
            not free for free in self.free_rod_nodes_indices
        ]
        initial_system_state = self.initialize_system_state()
        self.get_energy_traced = torch.jit.trace(self.get_energy, [Tensor([0]), initial_system_state.free_nodes, initial_system_state.breakages])


    def initialize_surface_energy(self) -> Tensor:
        """
        Initialize the surface energy.
        We add a small noise to the surface energy to break the symmetry.

        :return: Initialized surface energy
        """
        torch.manual_seed(self.seed)
        return torch.normal(
            self.config.C_surface_average,
            self.config.C_surface_std,
            (self.config.N - 1,),
        )

    def initialize_system_state(self) -> SystemState:
        """
        Initialize the positions of the rod.
        Initially, the rod is straight, with the nodes evenly spaced between 0 and W.
        The outer points

        :return: Initialized SystemState
        """
        x_positions = torch.linspace(0, self.config.W, self.config.N)
        y_positions = torch.zeros(self.config.N)
        free_nodes =  torch.stack([x_positions, y_positions], dim=1)[ 1:-1, :]
        breakages = torch.zeros(self.config.N - 1)
        return SystemState(free_nodes, breakages)

    def assemble_rod(self, free_nodes: Tensor, controlled_nodes: Tensor) -> Tensor:
        """
        Assemble the rod from the free and controlled nodes.

        :param free_nodes: Free nodes
        :param controlled_nodes: Controlled nodes
        :return: Assembled rod
        """
        res = torch.empty((len(self.controlled_nodes_indices), free_nodes.shape[1]), dtype=free_nodes.dtype)
        res[self.free_rod_nodes_indices] = free_nodes
        res[self.controlled_nodes_indices] = controlled_nodes
        return res
    def get_controlled_nodes_position(self, time: Tensor | float) -> Tensor:
        """
        Get the position of the controlled nodes at a given time.

        :param time: Time of shape (1,)
        :return: Position of the controlled nodes
        """
        if not torch.is_tensor(time):
            time = torch.tensor([time])
        first_position = torch.cat(
            [torch.zeros((1, 1)), time.view(1, 1) * self.config.d], dim=1
        )
        last_position = torch.cat(
            [torch.ones((1, 1)) * self.config.W, -time.view(1, 1) * self.config.d], dim=1
        )
        return torch.cat([first_position, last_position], dim=0)

    def compute_stress_vector(
        self, positions: Tensor, fracture_vector: Tensor
    ) -> Tensor:
        """
        Compute the stress vector.

        :param positions: Current positions of the full rod
        :param fracture_vector: Current fracture state
        :return: Computed stress vector
        """
        x_positions = positions[:, 0]
        y_positions = positions[:, 1]
        dx = x_positions[1:] - x_positions[:-1]
        dy = y_positions[1:] - y_positions[:-1]
        distances = torch.sqrt(dx**2 + dy**2)
        stress = self.config.k_stretch * (distances - self.config.L) ** 2
        weights = 1 - fracture_vector
        return stress * weights

    def compute_elastic_energy(
        self, positions: Tensor, fracture_vector: Tensor
    ) -> Tensor:
        """
        Compute the elastic energy.

        :param positions: Current positions of the full rod
        :param fracture_vector: Current fracture state
        :return: Computed elastic energy
        """
        stress = self.compute_stress_vector(positions, fracture_vector)
        return torch.sum(stress)

    def compute_surface_energy(self, fracture_vector: Tensor) -> Tensor:
        """
        Compute the surface energy.

        :param fracture_vector: Current fracture state
        :param surface_energy_constants: Surface energy constants
        :return: Computed surface energy
        """
        return torch.sum(fracture_vector * self.surface_energy_constants)

    def get_energy(self, time: Tensor, free_nodes: Tensor, breakages: Tensor) -> Tensor:
        """
        Compute the energy of the system.

        :param time: Time of shape (1)"
        :param free_nodes: Free nodes of shape (N-2, 2)
        :param breakages: Breakages of shape (N-1,)
        """
        controlled_nodes_position = self.get_controlled_nodes_position(time)
        rod = self.assemble_rod(free_nodes, controlled_nodes_position)
        return self.compute_surface_energy(
            breakages
        ) + self.compute_elastic_energy(rod, breakages)

    def transition_system_gradient_descent(
        self,
        time: Tensor,
        initial_system_state: SystemState,
    ) -> tuple[SystemState, Tensor, int]:
        """
        Let the system transition using gradient descent until it lands on a critical point.
        
        :param time: Time of shape (1,)
        :param initial_system_state: Initial system state
        :return: Final positions, final energy, and number of iterations
        """
        if not callable(self.get_energy_traced):
            raise TypeError("Somehow, the tracing went wrong")
        system_state = SystemState(
            initial_system_state.free_nodes.clone(),
            initial_system_state.breakages.clone(),
        )
        system_state.free_nodes.requires_grad_(True)
        system_state.breakages.requires_grad_(True)
        system_state.free_nodes.retain_grad()
        system_state.breakages.retain_grad()
        current_lr = self.config.learning_rate

        for iteration in tqdm(range(self.config.max_iterations), leave=False):
            with torch.no_grad():
                if system_state.free_nodes.grad is not None:
                    system_state.free_nodes.grad.zero_()
                if system_state.breakages.grad is not None:
                    system_state.breakages.grad.zero_()

            energy = self.get_energy_traced(time, system_state.free_nodes, system_state.breakages)
            energy.backward()  # type: ignore

            with torch.no_grad():
                if system_state.free_nodes.grad is None or system_state.breakages.grad is None:
                    raise TypeError("The gradients are None")
                gradient_norm = np.linalg.norm(
                    (system_state.free_nodes.grad).cpu().numpy().flatten(),
                    ord=self.config.error_norm_p,
                )
                system_state.free_nodes -= current_lr * system_state.free_nodes.grad
                system_state.breakages -= current_lr * system_state.breakages.grad
                system_state.breakages.clamp_(0, 1)
                fractor_gradient_finished = (system_state.breakages == 0) | (
                    system_state.breakages == 1
                )

                if fractor_gradient_finished.all() and (
                    gradient_norm < self.config.tolerance
                ):
                    break

            current_lr = max(
                current_lr * self.config.decay_factor, self.config.min_learning_rate
            )

        else:
            print("Maximum iterations reached without convergence.")
            print(f"Final Gradient Norm: {gradient_norm}")
            print(f"Fracture vector: {system_state.breakages}")

        return system_state, energy.detach(), iteration

    def calculate_energy_time_derivative(
            self,
            time: Tensor,
            system_state: SystemState,
    ) -> Tensor:
        """
        Calculate the energy time derivative.

        :param time: Time of shape (1,)
        :param system_state: System state
        :return: Energy time derivative
        """
        if not callable(self.get_energy_traced):
            raise TypeError("Somehow, the tracing went wrong")
        time.requires_grad_(True)
        
        if time.grad is not None:
            time.grad.zero_()
        energy = self.get_energy_traced(time, system_state.free_nodes, system_state.breakages)
        energy.backward()  # type: ignore
        return time.grad  # type: ignore

    def plot_simulation_step(
        self,
        system_state: SystemState,
        time: Tensor | float,
        energy: float,
        max_stress: float,
        ax: axes.Axes | None = None,
        fig: figure.Figure | None = None,
        cbar: colorbar.Colorbar | None = None,
        title: str | None = "Rod Configuration",
    ) -> Tuple[axes.Axes, colorbar.Colorbar]:
        """
        Plot the simulation step.

        :param system_state: System state
        :param time: Time of shape (1,)
        :param energy: Energy of the system
        :param max_stress: Maximum stress
        :param ax: Axes
        :param fig: Figure
        :param cbar: Colorbar
        :param title: Title of the plot
        :return: Axes and Colorbar
        """
        full_rod = self.assemble_rod(system_state.free_nodes, self.get_controlled_nodes_position(time))
        stress = self.compute_stress_vector(full_rod, system_state.breakages).detach().cpu().numpy()
        full_rod = full_rod.detach().cpu().numpy()
        if ax is None or fig is None:
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(12, 8))
            ax = self.ax
            fig = self.fig
        ax.clear()

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

        colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)


        for i in range(len(full_rod) - 1):
            if system_state.breakages[i] == 0:
                ax.plot(
                    [full_rod[i,0], full_rod[i + 1, 0]],
                    [full_rod[i, 1], full_rod[i + 1, 1]],
                    color=cmap(stress[i] / max_stress),
                    linewidth=2.5,
                    alpha=0.8,
                )

        ax.scatter(full_rod[:, 0], full_rod[:, 1], color="#2c3e50", s=10, zorder=5)

        if title:
            ax.set_title(
                title,
                fontweight="bold",
                fontsize=14,
            )

        ax.set_ylim(-self.config.d, self.config.d)

        if cbar is None:
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_stress)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label="Stress", pad=0.1)
            cbar.ax.set_ylabel("Stress", fontweight="bold")
            cbar.set_ticks([])


        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return ax, cbar

    
    def run_simulation(self, callback: Literal["plot","none"]="plot") -> list[SimulationStepResult]:
        """
        Run the simulation.

        :return: List of SimulationStepResult
        """
        
        plt.ion()
        ax, cbar = None, None
        time = torch.linspace(0, 1, self.config.time_steps)
        current_system_state = self.initialize_system_state()
        max_stress = torch.min(self.surface_energy_constants).item() * 1.2

        simulation_results = []
        iterator =  tqdm(time)
        for t in iterator:
            energy_before = self.get_energy(t, current_system_state.free_nodes, current_system_state.breakages)
            current_system_state, energy, iterations = self.transition_system_gradient_descent(
                t, current_system_state
            )
            iterator.set_description(f"Time: {t.item():.2f}, Iterations: {iterations}")
            energy_float = energy.item()
            energy_gradient = self.calculate_energy_time_derivative(t, current_system_state)
            simulation_results.append(
                SimulationStepResult(
                    energy_before_system_transition=energy_before.item(),
                    energy=energy_float,
                    energy_gradient=energy_gradient.item(),
                    SystemState=current_system_state,
                    time=t.item(),
                    iterations=iterations,
                )
            )
            if callback == "plot":
                ax, cbar = self.plot_simulation_step(
                    current_system_state, t, energy_float, max_stress, ax, self.fig, cbar
                )
                plt.draw()
                plt.pause(0.1)
        return simulation_results

    def fill_up_simulation(self, simulation_results: list[SimulationStepResult], factor: int):
        """
        Fill up the simulation results.

        :param simulation_results: Simulation results
        :param factor: Factor to fill up
        :return: Filled up simulation results
        """
        if not callable(self.get_energy_traced):
            raise TypeError("Somehow, the tracing went wrong")
        filled_simulation_results = []
        for i, result in enumerate(simulation_results):
            filled_simulation_results.append(result)
            if i < len(simulation_results) - 1:
                for j in range(1, factor):
                    time = result.time + j * (simulation_results[i + 1].time - result.time) / factor
                    new_energy = self.get_energy_traced(
                        torch.tensor([time]), result.SystemState.free_nodes, result.SystemState.breakages
                    )
                    new_energy_gradient = self.calculate_energy_time_derivative(
                        torch.tensor([time]), result.SystemState
                    )
                    filled_simulation_results.append(
                        SimulationStepResult(
                            energy_before_system_transition=new_energy.item(),
                            energy=new_energy.item(),
                            energy_gradient=new_energy_gradient.item(),
                            SystemState=result.SystemState,
                            time=time,
                            iterations=0,
                        )
                    )
        return filled_simulation_results



def read_config() -> dict[str, SimulationConfig]:
    """
    Read the configuration file.

    :return: Dictionary of simulation configurations
    """
    with open(Path(__file__).parent / "configuration.json") as file:
        content = json.load(file)
    return {name: SimulationConfig(**dct) for name, dct in content.items()}


RESULT_DIR = Path(__file__).parent / "simulation_results"


def do_all_simulations(overwrite: bool = False, plot: bool = True):
    """
    Run all simulations defined in the configuration file.

    :param overwrite: Whether to overwrite existing simulation results
    """
    RESULT_DIR.mkdir(exist_ok=True)
    configs = read_config()
    for i, (name, config) in enumerate(configs.items()):
        result_path = RESULT_DIR / name
        if (result_path).exists() and not overwrite:
            print(f"skipping {name}: already exists...")
            continue
        simulator = Simulator(config)
        print(f"Simulating {i}/{len(configs)}: {name}...")
        simulation_results = simulator.run_simulation(callback="plot" if plot else "none")
        with open(result_path, "wb") as file:
            pickle.dump(simulation_results, file)


def load_simulations() -> dict[str, list[SimulationStepResult]]:
    """
    Load all simulation results.

    :return: Dictionary of simulation results
    """
    def load_file(path: Path) -> list[SimulationStepResult]:
        with open(path, "rb") as file:
            return pickle.load(file)
    configs = read_config()
    return {name: load_file(RESULT_DIR / name) for name in configs.keys()}


def load_and_interpolate(total_frames: int) -> dict[str, list[SimulationStepResult]]:
    """
    Load and interpolate simulation results.

    :param total_frames: Total number of frames for interpolation
    :return: Dictionary of interpolated simulation results
    """
    simulation_results = load_simulations()
    configs = read_config()

    def get_interpolated(name: str) -> list[SimulationStepResult]:
        simulation_result = simulation_results[name]
        simulator = Simulator(config=configs[name])
        if total_frames % (len(simulation_result) - 1) != 0:
            raise ValueError("Total frames must be a multiple of the simulation length")
        return simulator.fill_up_simulation(
            simulation_result, factor=total_frames // (len(simulation_result) - 1)
        )

    return {name: get_interpolated(name) for name in configs.keys()}




def main():
    """
    Main function to run the simulations
    """
    do_all_simulations(overwrite=True, plot=False)


if __name__ == "__main__":
    main()
