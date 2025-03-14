import json
from pathlib import Path
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
from quasistatic_evol.rod_simulator import SimulationStepResult, load_and_interpolate, load_simulations, SystemState


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

def plot_energy_sums(results:list[SimulationStepResult], ax: axes.Axes | None = None):
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
    fig.savefig(output_path / "combined.svg")
    fig.savefig(output_path / "combined.png", dpi=300)
    plt.close(fig)

    # plot individual
    individual_fig = plt.figure(figsize=(12, 8))
    individual_ax = individual_fig.add_subplot(111)
    plotter.plot_energies(ax=individual_ax)
    individual_fig.savefig(output_path / "energies.svg")
    individual_fig.savefig(output_path / "energies.png", dpi=300)
    individual_ax.clear()
    plotter.plot_iteration_number(ax=individual_ax)
    individual_fig.savefig(output_path / "iterations.svg")
    individual_fig.savefig(output_path / "iterations.png", dpi=300)
    individual_ax.clear()
    plotter.plot_mu(ax=individual_ax)
    individual_fig.savefig(output_path / "mu.svg")
    individual_fig.savefig(output_path / "mu.png", dpi=300)
    individual_ax.clear()
    plotter.plot_d(ax=individual_ax)
    individual_fig.savefig(output_path / "d.svg")
    individual_fig.savefig(output_path / "d.png", dpi=300)
    plt.close(individual_fig)
    individual_ax.clear()
    plt.close(individual_fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plot_energy_sums(interpolated_simulations["15"], ax=axes[0])
    plot_energy_sums(interpolated_simulations["240"], ax=axes[1])
    fig.tight_layout()
    fig.savefig(output_path / "energy_sums.svg") 
    fig.savefig(output_path / "energy_sums.png", dpi=300)   
    plt.close(fig)

def main():
    OUTPUT_PATH = Path(__file__).parent / "plots"
    OUTPUT_PATH.mkdir(exist_ok=True)
    create_plots(OUTPUT_PATH)

if __name__ == "__main__":
    main()