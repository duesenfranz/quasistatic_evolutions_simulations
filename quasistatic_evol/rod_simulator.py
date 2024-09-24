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
from typing import Tuple


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
    tolerance: float = 5e-5
    max_iterations: int = 100000
    d: float = 5
    time_steps: int = 20
    plot_interval: int = N // 10
    error_norm_p: int = 2


@dataclass
class SimulationStepResult:
    energy_before_optimizing: float
    energy: float
    energy_gradient: float
    positions: Tensor
    breakages: Tensor
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

    def initialize_surface_energy(self) -> Tensor:
        """
        Initialize the surface energy.

        :return: Initialized surface energy
        """
        torch.manual_seed(self.seed)
        return torch.normal(
            self.config.C_surface_average,
            self.config.C_surface_std,
            (self.config.N - 1,),
        )

    def initialize_positions(self) -> Tensor:
        """
        Initialize the positions of the rod.

        :return: Initialized positions
        """
        x_positions = torch.linspace(0, self.config.W, self.config.N)
        y_positions = torch.zeros(self.config.N)
        return torch.stack([x_positions, y_positions])

    def initialize_fracture(self) -> Tensor:
        """
        Initialize the fracture vector.

        :return: Initialized fracture vector
        """
        return torch.zeros(self.config.N - 1)

    def compute_angles_from_positions(self, positions: Tensor) -> Tensor:
        """
        Compute angles from positions.

        :param positions: Current positions of the rod
        :return: Computed angles
        """
        x_positions = positions[0]
        y_positions = positions[1]
        dx = x_positions[1:] - x_positions[:-1]
        dy = y_positions[1:] - y_positions[:-1]
        return torch.atan2(dy, dx)

    def compute_surface_energy(
        self, fracture_vector: Tensor, surface_energy_constants: Tensor
    ) -> Tensor:
        """
        Compute the surface energy.

        :param fracture_vector: Current fracture state
        :param surface_energy_constants: Surface energy constants
        :return: Computed surface energy
        """
        return torch.sum(fracture_vector * surface_energy_constants)

    def compute_stress_vector(
        self, positions: Tensor, fracture_vector: Tensor
    ) -> Tensor:
        """
        Compute the stress vector.

        :param positions: Current positions of the rod
        :param fracture_vector: Current fracture state
        :return: Computed stress vector
        """
        x_positions = positions[0]
        y_positions = positions[1]
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

        :param positions: Current positions of the rod
        :param fracture_vector: Current fracture state
        :return: Computed elastic energy
        """
        stress = self.compute_stress_vector(positions, fracture_vector)
        return torch.sum(stress)

    def total_energy(
        self,
        positions: Tensor,
        fracture_vector: Tensor,
        surface_energy_constants: Tensor,
    ) -> Tensor:
        """
        Compute the total energy of the system.

        :param positions: Current positions of the rod
        :param fracture_vector: Current fracture state
        :param surface_energy_constants: Surface energy constants
        :return: Total energy of the system
        """
        surface_energy = self.compute_surface_energy(
            fracture_vector, surface_energy_constants
        )
        elastic_energy = self.compute_elastic_energy(positions, fracture_vector)
        return surface_energy + elastic_energy

    def plot_rod(
        self,
        simulation_step_result: SimulationStepResult,
        step: int,
        max_stress: float,
        ax: axes.Axes | None = None,
        fig: figure.Figure | None = None,
        cbar: colorbar.Colorbar | None = None,
    ) -> tuple[axes.Axes, colorbar.Colorbar | None]:
        """
        Plot the rod configuration.

        :param simulation_step_result: Result of the current simulation step
        :param step: Current step number
        :param max_stress: Maximum stress for color scaling
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param cbar: Existing colorbar (optional)
        :return: Updated axes and colorbar
        """
        x_positions = simulation_step_result.positions[0].detach().cpu().numpy()
        y_positions = simulation_step_result.positions[1].detach().cpu().numpy()
        stress = (
            self.compute_stress_vector(
                simulation_step_result.positions, simulation_step_result.breakages
            )
            .detach()
            .cpu()
            .numpy()
        )

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

        for i in range(len(x_positions) - 1):
            if simulation_step_result.breakages[i] == 0:
                ax.plot(
                    [x_positions[i], x_positions[i + 1]],
                    [y_positions[i], y_positions[i + 1]],
                    color=cmap(stress[i] / max_stress),
                    linewidth=2.5,
                    alpha=0.8,
                )

        ax.scatter(x_positions, y_positions, color="#2c3e50", s=30, zorder=5)

        ax.set_xlabel("X Position", fontweight="bold")
        ax.set_ylabel("Y Position", fontweight="bold")
        ax.set_title(
            f"Rod Configuration at Step {step}, Iteration {simulation_step_result.iterations}",
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

        plt.tight_layout()

        return ax, cbar

    def optimize_energy(
        self,
        positions: Tensor,
        fracture_vector: Tensor,
        surface_energy_constants: Tensor,
        position_mask: Tensor,
        step: int,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Optimize the energy of the system.

        :param positions: Current positions of the rod
        :param fracture_vector: Current fracture state
        :param surface_energy_constants: Surface energy constants
        :param position_mask: Mask for fixed positions
        :param step: Current step number
        :return: Optimized positions, final energy, and number of iterations
        """
        total_energy_fct = torch.jit.trace(
            self.total_energy, (positions, fracture_vector, surface_energy_constants)
        )
        if not callable(total_energy_fct):
            raise TypeError("Somehow, the tracing went wrong")
        positions.requires_grad_(True)
        fracture_vector.requires_grad_(True)
        current_lr = self.config.learning_rate

        for iteration in tqdm(range(self.config.max_iterations), leave=False):
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad.zero_()
                if fracture_vector.grad is not None:
                    fracture_vector.grad.zero_()

            energy = total_energy_fct(
                positions, fracture_vector, surface_energy_constants
            )
            energy.backward()

            with torch.no_grad():
                gradient_norm = np.linalg.norm(
                    (positions.grad * (~position_mask)).cpu().numpy().flatten(),
                    ord=self.config.error_norm_p,
                )
                if positions.grad is None or fracture_vector.grad is None:
                    raise TypeError("The gradients are None")
                positions -= current_lr * positions.grad * (~position_mask)
                fracture_vector -= current_lr * fracture_vector.grad
                positions.data[position_mask] = positions[position_mask]
                fracture_vector.clamp_(0, 1)
                fractor_gradient_finished = (fracture_vector == 0) | (
                    fracture_vector == 1
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
            print(f"Fracture vector: {fracture_vector}")

        return positions.detach(), energy.detach(), iteration

    def calculate_energy_derivative(
        self,
        position: Tensor,
        fracture_vector: Tensor,
        surface_energy_constants: Tensor,
        time: float,
    ) -> float:
        """
        Calculate the energy derivative.

        :param position: Current positions of the rod
        :param fracture_vector: Current fracture state
        :param surface_energy_constants: Surface energy constants
        :param time: Current time
        :return: Energy derivative
        """
        def calculate_energy(time: Tensor) -> Tensor:
            first_position = torch.cat(
                [torch.zeros((1, 1)), time.view(1, 1) * self.config.d], dim=0
            )
            last_position = torch.cat(
                [torch.zeros((1, 1)), -time.view(1, 1) * self.config.d], dim=0
            )
            positions = torch.cat(
                [first_position, position[:, 1:-1], last_position], dim=1
            )
            energy = self.total_energy(
                positions=positions,
                fracture_vector=fracture_vector,
                surface_energy_constants=surface_energy_constants,
            )
            return energy

        time_tensor = torch.from_numpy(np.array(time))
        time_tensor.requires_grad_(True)
        energy = calculate_energy(time_tensor)
        energy.backward()
        if time_tensor.grad is None:
            raise TypeError("The gradient is None")
        return time_tensor.grad.item()

    def run_simulation(self) -> list[SimulationStepResult]:
        """
        Run the complete simulation.

        :return: List of simulation step results
        """
        plt.ion()
        ax, cbar = None, None
        positions = self.initialize_positions()
        fracture_vector = self.initialize_fracture()
        surface_energy_constants = self.initialize_surface_energy()
        max_stress = torch.min(surface_energy_constants).item() * 1.2

        simulation_step_results: list[SimulationStepResult] = []

        for n in tqdm(range(self.config.time_steps)):
            with torch.no_grad():
                positions[1, 0] = (n) * self.config.d / self.config.time_steps
                positions[1, -1] = -(n) * self.config.d / self.config.time_steps

            position_mask = torch.zeros_like(positions, dtype=torch.bool)
            position_mask[:, 0] = True
            position_mask[:, -1] = True
            original_energy = self.total_energy(
                positions=positions,
                fracture_vector=fracture_vector,
                surface_energy_constants=surface_energy_constants,
            )
            energy_derivative = self.calculate_energy_derivative(
                position=positions,
                fracture_vector=fracture_vector,
                surface_energy_constants=surface_energy_constants,
                time=n / self.config.time_steps,
            )

            optimized_positions, final_energy, iterations = self.optimize_energy(
                positions, fracture_vector, surface_energy_constants, position_mask, n
            )

            result = SimulationStepResult(
                energy_gradient=energy_derivative,
                energy_before_optimizing=original_energy.item(),
                energy=final_energy.item(),
                iterations=iterations,
                positions=optimized_positions.clone(),
                breakages=fracture_vector.clone(),
            )
            ax, cbar = self.plot_rod(result, n, max_stress, ax, self.fig, cbar)
            plt.draw()
            plt.pause(0.2)
            simulation_step_results.append(result)

        plt.ioff()

        return simulation_step_results

    def calculate_ideal_breakage_time(self) -> int:
        """
        Calculate the ideal breakage time.

        :return: Ideal breakage time
        """
        minimal_constant = torch.min(self.initialize_surface_energy()).item()
        displacement_for_breakage = np.sqrt(minimal_constant / self.config.k_stretch)
        length_per_segment_for_breakage = self.config.L + displacement_for_breakage
        total_length = self.config.N * length_per_segment_for_breakage
        y_displacement = np.sqrt(total_length**2 - self.config.W**2)
        breakage_time = y_displacement / (2 * self.config.d)
        return breakage_time

    def interpolate_piecewise_static(
        self, simulation_step_results: list[SimulationStepResult], n_frames: int
    ) -> list[SimulationStepResult]:
        """
        Interpolate the simulation results for smooth animation.

        :param simulation_step_results: List of simulation step results
        :param n_frames: Number of frames for interpolation
        :return: List of interpolated simulation step results
        """
        surface_energy = self.initialize_surface_energy()

        def get_new_animation_step(i: int) -> SimulationStepResult:
            animation_step_to_take = int(i / n_frames * len(simulation_step_results))
            next_animation_step_to_take = int(
                (i + 1) / n_frames * len(simulation_step_results)
            )
            relevant_result = simulation_step_results[animation_step_to_take]
            positions = relevant_result.positions.clone()
            positions[1, 0] = (i) * self.config.d / n_frames
            positions[1, -1] = -(i) * self.config.d / n_frames
            energy = self.total_energy(
                positions, relevant_result.breakages, surface_energy
            )
            if next_animation_step_to_take != animation_step_to_take:
                original_energy = relevant_result.energy_before_optimizing
            else:
                original_energy = energy.item()
            energy_derivative = self.calculate_energy_derivative(
                position=positions,
                fracture_vector=relevant_result.breakages,
                surface_energy_constants=surface_energy,
                time=i / n_frames,
            )
            return SimulationStepResult(
                energy_gradient=energy_derivative,
                energy_before_optimizing=original_energy,
                energy=energy.item(),
                positions=positions,
                breakages=relevant_result.breakages,
                iterations=relevant_result.iterations,
            )

        return [get_new_animation_step(i) for i in range(n_frames)]

    @classmethod
    def plot_energies(
        cls,
        simulation_step_results: dict[str, list[SimulationStepResult]],
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
        color_map: dict[str, str] | None = None
    ) -> axes.Axes:
        """
        Plot the energy over time for multiple simulations.

        :param simulation_step_results: Dictionary of simulation results
        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
        :param color_map: Dictionary mapping simulation names to colors
        :return: Updated axes and figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.clear()

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\mathcal{E}_t(u^\delta)$")
        ax.set_title("Total Energy Over Time")
        for name, steps in simulation_step_results.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = color_map[name] if color_map and name in color_map else None
            ax.plot(x_data, [result.energy for result in steps], label=r"$\delta = \frac{1}{" + name + r"}$", color=color)
        if add_legend:
            ax.legend()
        return ax

    @classmethod
    def plot_iteration_number(
        cls,
        simulation_step_results: dict[str, list[SimulationStepResult]],
        start_time: float = 0,
        end_time: float = 1,
        x_resolution: int = 200,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
        color_map: dict[str, str] | None = None
    ) -> axes.Axes:
        """
        Plot the number of iterations over time for multiple simulations.

        :param simulation_step_results: Dictionary of simulation results
        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param x_resolution: Resolution of the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
        :param color_map: Dictionary mapping simulation names to colors
        :return: Updated axes and figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        ax.clear()

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Number of Iterations")
        ax.set_title("Number of Iterations Until Convergence")
        for name, steps in simulation_step_results.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = color_map[name] if color_map and name in color_map else None
            ax.plot(
                x_data,
                [result.iterations for result in steps],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                drawstyle="steps-mid",
                marker="o",
                markersize=50 / len(steps),
                color=color
            )
        if add_legend:
            ax.legend()
        return ax

    @classmethod
    def plot_mu(
        cls,
        simulation_step_results: dict[str, list[SimulationStepResult]],
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
        color_map: dict[str, str] | None = None
    ) -> axes.Axes:
        """
        Plot the mu values over time for multiple simulations.

        :param simulation_step_results: Dictionary of simulation results
        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
        :param color_map: Dictionary mapping simulation names to colors
        :return: Updated axes and figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        ax.clear()

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\mu^\delta(\{t\})$")
        ax.set_title(r"$\mu^\delta$")
        for name, steps in simulation_step_results.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = color_map[name] if color_map and name in color_map else None
            ax.vlines(
                x_data,
                ymax=0,
                ymin=[result.energy - result.energy_before_optimizing for result in steps],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                colors=color,
            )
        if add_legend:
            ax.legend()
        return ax

    @classmethod
    def plot_d(
        cls,
        simulation_step_results: dict[str, list[SimulationStepResult]],
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
        color_map: dict[str, str] | None = None
    ) -> axes.Axes:
        """
        Plot the d values over time for multiple simulations.

        :param simulation_step_results: Dictionary of simulation results
        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
        :param color_map: Dictionary mapping simulation names to colors
        :return: Updated axes and figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        ax.clear()

        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\mathcal{D}^\delta(t)$")
        ax.set_title(r"$\mathcal{D}^\delta$")
        for name, steps in simulation_step_results.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = color_map[name] if color_map and name in color_map else None
            ax.plot(x_data, [result.energy_gradient for result in steps], label=r"$\delta = \frac{1}{" + name + r"}$", color=color)
        if add_legend:
            ax.legend()
        return ax



def read_config() -> dict[str, SimulationConfig]:
    """
    Read the configuration file.

    :return: Dictionary of simulation configurations
    """
    with open(Path(__file__).parent / "configuration.json") as file:
        content = json.load(file)
    return {name: SimulationConfig(**dct) for name, dct in content.items()}


RESULT_DIR = Path(__file__).parent / "simulation_results"
VIDEO_DIR = Path(__file__).parent / "videos"


def do_all_simulations(overwrite: bool = False):
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
        simulation_results = simulator.run_simulation()
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
        return simulator.interpolate_piecewise_static(
            simulation_result, n_frames=total_frames
        )

    return {name: get_interpolated(name) for name in configs.keys()}


def create_videos(fps: int = 60, clip_length: float = 5, overwrite: bool = False):
    """
    Create videos of the simulations.

    :param fps: Frames per second for the videos
    :param clip_length: Length of the video clips in seconds
    :param overwrite: Whether to overwrite existing videos
    """
    VIDEO_DIR.mkdir(exist_ok=True)
    interpolated_frames = load_and_interpolate(int(fps * clip_length))
    configs = read_config()
    for name, results in interpolated_frames.items():
        output_path = VIDEO_DIR / f"{name}.mp4"
        if output_path.exists() and not overwrite:
            print(f"Skipping {name}: already exists")
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        simulator = Simulator(configs[name])
        max_stress = torch.min(simulator.initialize_surface_energy()).item() * 1.2

        _, cbar = simulator.plot_rod(results[0], 0, max_stress, ax, fig)
        progress_bar = tqdm(
            total=fps * clip_length, desc=f"Creating video {name}", unit="frame"
        )

        def animate(i):
            simulator.plot_rod(
                results[i],
                0,
                max_stress,
                ax,
                fig,
                cbar,
            )
            progress_bar.update(1)
            return [ax]

        try:
            anim = animation.FuncAnimation(
                fig,
                animate,
                frames=int(fps * clip_length),
                interval=200,
                blit=False,
            )
            anim.save(str(output_path), writer="ffmpeg", fps=fps)
        finally:
            progress_bar.close()


def create_plots():
    """
    Create plots of the simulation results.
    """
    OUTPUT_PATH = Path(__file__).parent / "plots"
    OUTPUT_PATH.mkdir(exist_ok=True)
    simulations = load_simulations()
    interpolated_simulations = load_and_interpolate(200)

    color_map = {name: color for name, color in zip(simulations.keys(), plt.get_cmap('tab10').colors)}

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))

    def plot_and_save(ax, plot_func, data, filename):
        plot_func(data, ax=ax, color_map=color_map)
        individual_fig = plt.figure(figsize=(12, 8))
        individual_ax = individual_fig.add_subplot(111)
        plot_func(data, ax=individual_ax, color_map=color_map)
        individual_fig.savefig(OUTPUT_PATH / f"{filename}.svg")
        individual_fig.savefig(OUTPUT_PATH / f"{filename}.png")
        plt.close(individual_fig)

    plot_and_save(ax1, Simulator.plot_energies, interpolated_simulations, "energies")
    plot_and_save(ax2, Simulator.plot_iteration_number, simulations, "iterations")
    plot_and_save(ax3, Simulator.plot_mu, simulations, "mu")
    plot_and_save(ax4, Simulator.plot_d, interpolated_simulations, "d")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / "combined.svg")
    plt.close(fig)


def main():
    """
    Main function to run the simulations, create videos, and plots.
    """
    do_all_simulations(overwrite=False)
    create_videos()
    create_plots()


if __name__ == "__main__":
    main()
