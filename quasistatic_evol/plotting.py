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
import scienceplots



# get those like here: https://en.wikibooks.org/wiki/LaTeX/Page_Layout
@dataclass
class PlotConfig:
    line_width: float = 5.8055556
    font_size: float = 10
    font_size_small: float = 8
    font_size_large: float = 10
    font_family: str = "cm"
    


class Plotter:
    def __init__(
        self,
        simulations: dict[str, list[SimulationStepResult]],
        simulators: dict[str, Simulator],
        color_map: dict[str, str] | None,
        plot_config: PlotConfig = PlotConfig(),
    ) -> None:
        self.simulations = simulations
        self.simulators = simulators
        self.color_map = color_map or {
            name: color
            for name, color in zip(simulations.keys(), plt.get_cmap("tab10").colors)  # type: ignore
        }
        self.plot_config = plot_config
        plt.style.use('science')
        sns.set_style("whitegrid")
        sns.set_context("paper")
        plt.rc('font', size=self.plot_config.font_size )          # controls default text sizes
        plt.rc('axes', titlesize=self.plot_config.font_size)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.plot_config.font_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.plot_config.font_size_small)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.plot_config.font_size_small)    # fontsize of the tick labels
        plt.rc('legend', fontsize=self.plot_config.font_size_small)    # legend fontsize
        plt.rc('figure', titlesize=self.plot_config.font_size)  # fontsize of the figure title
        plt.rc('legend', framealpha=0.8, frameon=True)
        plt.rc('text', usetex=True)
        plt.rc('mathtext', fontset='cm')
        plt.rc('mathtext', rm='cm')
        plt.rc('mathtext', it='cm:italic')
        plt.rc('mathtext', bf='cm:bold')
        plt.rc('font', family='cm')
        plt.rc('font',**{'family':'serif','serif':[plot_config.font_family]})

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
            fig, ax = plt.subplots(figsize=self.get_figsize(0.75))

        ax.clear()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\\mathcal{E}_t(u^\\delta)$")
        ax.set_title("Total Energy Over Time",)
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

    def get_figsize(self, height_to_width_ratio=0.75, scaling: float=1) -> tuple[float, float]:
        return (self.plot_config.line_width * scaling, self.plot_config.line_width * height_to_width_ratio * scaling)

    def plot_iteration_number(
        self,
        start_time: float = 0,
        end_time: float = 1,
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
            fig, ax = plt.subplots(figsize=self.get_figsize(0.75))
        ax.clear()
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
            fig, ax = plt.subplots(figsize=self.get_figsize(0.75))
        ax.clear()
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
            ax.legend(loc='lower left')
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


    def plot_energy_sums(self, name: str, ax: axes.Axes | None = None):
        """
        Plot the sum of the energies over time for a single simulation
        """
        if ax is None:
                fig, ax = plt.subplots(figsize=self.get_figsize(0.75))
        ax.clear()
        ax.set_xlabel("$t$")
        
        results = self.simulations[name]
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
        self
    ) -> Figure:
        """
        Plot the key frames of a simulation.
        This means the first frame, the frame right before breakage, the one right after breakage
        and the last frame.
        """
        # fig, all_axes = plt.subplots(len(simulations), 4, figsize=(14, 4))
        fig = plt.figure(figsize=self.get_figsize(0.27 * len(self.simulations)))
        gs = gridspec.GridSpec(len(self.simulations), 5, width_ratios=[1, 1, 1, 1, 0.05])
        all_axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(len(self.simulations))]
        cbar_ax = fig.add_subplot(gs[:, 4])
        max_stress = torch.min(list(self.simulators.values())[0].initialize_surface_energy()).item() * 1.2
        cbar = None
        for letter, (name, results), axes in zip(string.ascii_lowercase, self.simulations.items(), all_axes):
            mu = [result.energy_before_system_transition - result.energy for result in results]
            max_mu_index = np.argmax(mu)
            before_breakage = results[max(0, max_mu_index - 1)]  # type: ignore
            after_breakage = results[max_mu_index]
            for ax, result in zip(
                axes,
                [results[0], before_breakage, after_breakage, results[-1]],
            ):
                _, cbar = self.simulators[name].plot_simulation_step(
                    result.SystemState,
                    result.time,
                    result.energy,
                    max_stress,
                    ax=ax,
                    fig=fig,
                    cbar=cbar,
                    title=None,
                    cax=cbar_ax,
                    point_size=0.3
                )
            axes[0].set_ylabel(f"({letter})", fontsize=self.plot_config.font_size, labelpad=15, rotation=0)
        for ax, col_label in zip(all_axes[0], ["(i)", "(ii)", "(iii)", "(iv)"]):
            ax.set_title(col_label, pad=10, fontsize=self.plot_config.font_size)
        
        fig.tight_layout()
        return fig

def save_three_formats(fig: Figure, path: Path):

    fig.tight_layout()
    # fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))

def create_plots(output_path: Path):
    """
    Create plots of the simulation results.
    """
    interpolated_simulations = load_and_interpolate(96 * 5)
    configs = read_config()
    simulators = {name: Simulator(config) for name, config in configs.items()}
    plotter = Plotter(interpolated_simulations, simulators, None)

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
    individual_fig = plt.figure(figsize=plotter.get_figsize(0.75, scaling=0.5))
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

    n_of_configs = len(interpolated_simulations)
    fig, axes = plt.subplots(1, n_of_configs, figsize=plotter.get_figsize(0.5), sharey=True)
    for i, name in enumerate(interpolated_simulations.keys()):
        plotter.plot_energy_sums(name, ax=axes[i])
    fig.tight_layout()
    save_three_formats(fig, output_path / "energy_sums")
    plt.close(fig)

    key_frames_fig = plotter.plot_key_frames()
    save_three_formats(key_frames_fig, output_path / "key_frames")
    plt.close(key_frames_fig)


def main():
    OUTPUT_PATH = Path(__file__).parent / "plots"
    OUTPUT_PATH.mkdir(exist_ok=True)
    create_plots(OUTPUT_PATH)


if __name__ == "__main__":
    main()
