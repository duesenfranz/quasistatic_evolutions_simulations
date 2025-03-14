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

from quasistatic_evol.rod_simulator import Simulator, load_and_interpolate, read_config, SimulationStepResult, SystemState

VIDEO_DIR = Path(__file__).parent / "videos"


def create_videos(fps: int = 96, clip_length: float = 5, overwrite: bool = False):
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

        _, cbar = simulator.plot_simulation_step(results[0].SystemState, results[0].time, results[0].energy, max_stress, ax, fig, cbar=None)
        progress_bar = tqdm(
            total=fps * clip_length, desc=f"Creating video {name}", unit="frame"
        )

        def animate(i):
            simulator.plot_simulation_step(
                results[i].SystemState, results[i].time, results[i].energy,
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

def main():
    create_videos()

if __name__ == "__main__":
    main()