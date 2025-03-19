import json
from pathlib import Path
import string
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import axes, gridspec
from matplotlib import figure
from matplotlib.figure import Figure
from matplotlib import colorbar
import torch
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import os
from typing import Literal, Tuple
from quasistatic_evol.rod_simulator import (
    SimulationStepResult,
    Simulator,
    load_and_interpolate,
    load_simulations,
    SystemState,
    read_config,
)


class Plotter:
    def __init__(
        self,
        simulations: dict[str, list[SimulationStepResult]],
        color_map: dict[str, str] | None,
    ) -> None:
        self.simulations = simulations
        self.color_map = color_map or {
            name: color
            for name, color in zip(simulations.keys(), plt.get_cmap("tab10").colors)  # type: ignore
        }

    def plot_energies(
        self,
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
    ) -> axes.Axes:
        """
        Plot the energy over time for multiple simulations.

        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
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
        for name, steps in self.simulations.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = self.color_map[name]
            ax.plot(
                x_data,
                [result.energy for result in steps],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                color=color,
            )
        if add_legend:
            ax.legend()
        return ax

    def plot_iteration_number(
        self,
        start_time: float = 0,
        end_time: float = 1,
        x_resolution: int = 200,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
    ) -> axes.Axes:
        """
        Plot the number of iterations over time for multiple simulations.

        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param x_resolution: Resolution of the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
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
        for name, steps in self.simulations.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = self.color_map[name]
            ax.plot(
                x_data,
                [result.iterations for result in steps],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                drawstyle="steps-mid",
                marker="o",
                markersize=50 / len(steps),
                color=color,
            )
        if add_legend:
            ax.legend()
        return ax

    def plot_mu(
        self,
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
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
        for name, steps in self.simulations.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = self.color_map[name]
            ax.vlines(
                x_data,
                ymax=0,
                ymin=[
                    result.energy - result.energy_before_system_transition
                    for result in steps
                ],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                colors=color,
            )
        if add_legend:
            ax.legend()
        return ax

    def plot_d(
        self,
        start_time: float = 0,
        end_time: float = 1,
        ax: axes.Axes | None = None,
        add_legend: bool = True,
    ) -> axes.Axes:
        """
        Plot the d values over time for multiple simulations.

        :param start_time: Start time for the x-axis
        :param end_time: End time for the x-axis
        :param ax: Matplotlib axes to plot on (optional)
        :param fig: Matplotlib figure to plot on (optional)
        :param add_legend: Whether to add a legend to the plot
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
        for name, steps in self.simulations.items():
            x_data = np.linspace(start_time, end_time, len(steps), endpoint=False)
            color = self.color_map[name]
            ax.plot(
                x_data,
                [result.energy_gradient for result in steps],
                label=r"$\delta = \frac{1}{" + name + r"}$",
                color=color,
            )
        if add_legend:
            ax.legend()
        return ax


def plot_energy_sums(results: list[SimulationStepResult], ax: axes.Axes | None = None):
    """
    Plot the sum of the energies over time for a single simulation
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    ax.clear()

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    ax.set_xlabel("$t$")

    # reduce the mus by summing over them
    def get_mu_sum():
        current_sum = 0
        for result in results:
            current_sum += result.energy_before_system_transition - result.energy
            yield current_sum

    mu_sums = np.array(list(get_mu_sum()))
    energies_after = np.array([result.energy for result in results])
    integral = energies_after + mu_sums
    x_data = np.array([result.time for result in results])
    ax.plot(x_data, integral, label="$\int_0^t \mathcal{D}^\delta(s)\,\mathrm{d}s$")
    ax.plot(x_data, energies_after, label="$E_t(u^\delta(t))$")
    ax.legend()
    return ax


def plot_key_frames(
    simulations: dict[str, list[SimulationStepResult]], simulators: dict[str, Simulator]
) -> Figure:
    """
    Plot the key frames of a simulation.
    This means the first frame, the frame right before breakage, the one right after breakage
    and the last frame.
    """
    # fig, all_axes = plt.subplots(len(simulations), 4, figsize=(14, 4))
    fig = plt.figure(figsize=(14, 4 * len(simulations)))
    gs = gridspec.GridSpec(len(simulations), 5, width_ratios=[1, 1, 1, 1, 0.05])
    all_axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(len(simulations))]
    cbar_ax = fig.add_subplot(gs[:, 4])
    max_stress = torch.min(list(simulators.values())[0].initialize_surface_energy()).item() * 1.2
    cbar = None
    for letter, (name, results), axes in zip(string.ascii_lowercase, simulations.items(), all_axes):
        mu = [result.energy_before_system_transition - result.energy for result in results]
        max_mu_index = np.argmax(mu)
        before_breakage = results[max(0, max_mu_index - 1)]  # type: ignore
        after_breakage = results[max_mu_index]
        for ax, result in zip(
            axes,
            [results[0], before_breakage, after_breakage, results[-1]],
        ):
            _, cbar = simulators[name].plot_simulation_step(
                result.SystemState,
                result.time,
                result.energy,
                max_stress,
                ax=ax,
                fig=fig,
                cbar=cbar,
                title=None,
                cax=cbar_ax,
            )
        axes[0].set_ylabel(f"({letter})", fontsize=12, labelpad=15, rotation=0)
    for ax, col_label in zip(all_axes[0], ["(i): Initial", "(ii): Before Breakage", "(iii): After Breakage", "(iv): Final"]):
        ax.set_title(col_label, fontsize=12, pad=10)
    
    fig.tight_layout()
    return fig

def save_three_formats(fig: Figure, path: Path):
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))

def create_plots(output_path: Path):
    """
    Create plots of the simulation results.
    """
    interpolated_simulations = load_and_interpolate(96 * 5)
    plotter = Plotter(interpolated_simulations, None)

    # plot combined
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
    plotter.plot_energies(ax=ax1)
    plotter.plot_iteration_number(ax=ax2)
    plotter.plot_mu(ax=ax3)
    plotter.plot_d(ax=ax4)
    fig.tight_layout()
    save_three_formats(fig, output_path / "combined")
    plt.close(fig)

    # plot individual
    individual_fig = plt.figure(figsize=(12, 8))
    individual_ax = individual_fig.add_subplot(111)
    plotter.plot_energies(ax=individual_ax)
    save_three_formats(individual_fig, output_path / "energies")
    individual_ax.clear()
    plotter.plot_iteration_number(ax=individual_ax)
    save_three_formats(individual_fig, output_path / "iterations")
    individual_ax.clear()
    plotter.plot_mu(ax=individual_ax)
    save_three_formats(individual_fig, output_path / "mu")
    individual_ax.clear()
    plotter.plot_d(ax=individual_ax)
    save_three_formats(individual_fig, output_path / "d")
    plt.close(individual_fig)
    individual_ax.clear()
    plt.close(individual_fig)

    n_of_configs = len(interpolated_simulations)
    fig, axes = plt.subplots(1, n_of_configs, figsize=(10, 5), sharey=True)
    for i, result in enumerate(interpolated_simulations.values()):
        plot_energy_sums(result, ax=axes[i])
    fig.tight_layout()
    save_three_formats(fig, output_path / "energy_sums")
    plt.close(fig)

    configs = read_config()
    # for name, results in interpolated_simulations.items():
    #     simulator = Simulator(configs[name])
    #     key_frames_fig = plot_key_frames(results, simulator)
    #     save_three_formats(key_frames_fig, output_path / f"key_frames_{name}")
    #     plt.close(key_frames_fig)
    simulators = {name: Simulator(config) for name, config in configs.items()}
    key_frames_fig = plot_key_frames(interpolated_simulations, simulators)
    save_three_formats(key_frames_fig, output_path / "key_frames")
    plt.close(key_frames_fig)


def main():
    OUTPUT_PATH = Path(__file__).parent / "plots"
    OUTPUT_PATH.mkdir(exist_ok=True)
    create_plots(OUTPUT_PATH)


if __name__ == "__main__":
    main()
