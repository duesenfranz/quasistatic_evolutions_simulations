import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

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

class Simulator:
    def __init__(self, config: SimulationConfig, seed: int = 0):
        self.config = config
        self.seed = seed
        self.fig = None
        self.ax = None
        plt.ion()

    def initialize_surface_energy(self) -> Tensor:
        torch.manual_seed(self.seed)
        return torch.normal(self.config.C_surface_average, self.config.C_surface_std, (self.config.N - 1,))

    def initialize_positions(self) -> Tensor:
        x_positions = torch.linspace(0, self.config.W, self.config.N)
        y_positions = torch.zeros(self.config.N)
        return torch.stack([x_positions, y_positions])

    def initialize_fracture(self) -> Tensor:
        return torch.zeros(self.config.N - 1)

    def compute_angles_from_positions(self, positions: Tensor) -> Tensor:
        x_positions = positions[0]
        y_positions = positions[1]
        dx = x_positions[1:] - x_positions[:-1]
        dy = y_positions[1:] - y_positions[:-1]
        return torch.atan2(dy, dx)

    def compute_surface_energy(self, fracture_vector: Tensor, surface_energy_constants: Tensor) -> Tensor:
        return torch.sum(fracture_vector * surface_energy_constants)

    def compute_stress_vector(self, positions: Tensor, fracture_vector: Tensor) -> Tensor:
        x_positions = positions[0]
        y_positions = positions[1]
        dx = x_positions[1:] - x_positions[:-1]
        dy = y_positions[1:] - y_positions[:-1]
        distances = torch.sqrt(dx**2 + dy**2)
        stress = self.config.k_stretch * (distances - self.config.L) ** 2
        weights = 1 - fracture_vector
        return stress * weights

    def compute_elastic_energy(self, positions: Tensor, fracture_vector: Tensor) -> Tensor:
        stress = self.compute_stress_vector(positions, fracture_vector)
        return torch.sum(stress)

    def total_energy(self, positions: Tensor, fracture_vector: Tensor, surface_energy_constants: Tensor) -> Tensor:
        surface_energy = self.compute_surface_energy(fracture_vector, surface_energy_constants)
        elastic_energy = self.compute_elastic_energy(positions, fracture_vector)
        return surface_energy + elastic_energy

    def plot_stick(self, positions: Tensor, fracture_vector: Tensor, iteration: int, step: int, max_stress: float):
        global fig, ax
        
        x_positions = positions[0].detach().cpu().numpy()
        y_positions = positions[1].detach().cpu().numpy()

        # Compute stress vector
        stress = self.compute_stress_vector(positions, fracture_vector).detach().cpu().numpy()

        if self.fig is None or self.ax is None:
            # Create a new figure and axis if they don't exist
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
        else:
            # Clear the existing plot
            self.ax.clear()
        
        # Create a colormap
        cmap = plt.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=0, vmax=max_stress)

        # Plot segments with color based on stress
        for i in range(len(x_positions) - 1):
            if fracture_vector[i] == 0:
                self.ax.plot([x_positions[i], x_positions[i+1]], 
                        [y_positions[i], y_positions[i+1]], 
                        color=cmap(norm(stress[i])),
                        linewidth=2)

        # Plot points
        self.ax.plot(x_positions, y_positions, 'ko', markersize=0.5)

        self.ax.set_title(f"Stick Configuration at Step {step}, Iteration {iteration}")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.grid(True)
        # Set the y-axis limits to always range from -d to d
        self.ax.set_ylim(-self.config.d, self.config.d)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if len(self.ax.figure.axes) <= 1:
            plt.colorbar(sm, ax=self.ax, label='Stress')

        # This allows the plot to update without blocking
        plt.draw()
        plt.pause(0.2)  # Pause briefly to allow the figure to update

    def optimize_energy(self, positions: Tensor, fracture_vector: Tensor, surface_energy_constants: Tensor, position_mask: Tensor, step: int) -> tuple[Tensor, Tensor, int]:
        max_stress = torch.min(surface_energy_constants).item() * 1.2

        total_energy_fct = torch.jit.trace(self.total_energy, (positions, fracture_vector, surface_energy_constants))
        positions.requires_grad_(True)
        fracture_vector.requires_grad_(True)
        current_lr = self.config.learning_rate

        for iteration in tqdm(range(self.config.max_iterations), leave=False):
            energy = total_energy_fct(positions, fracture_vector, surface_energy_constants)
            energy.backward()

            with torch.no_grad():
                gradient_norm = np.linalg.norm((positions.grad * (~position_mask)).cpu().numpy().flatten(), ord=self.config.error_norm_p)
                positions -= current_lr * positions.grad * (~position_mask)
                positions.grad.zero_()
                fracture_vector -= current_lr * fracture_vector.grad
                fracture_vector.grad.zero_()
                positions.data[position_mask] = positions[position_mask]
                fracture_vector.clamp_(0, 1)
                fractor_gradient_finished = (fracture_vector == 0) | (fracture_vector == 1)

                if fractor_gradient_finished.all() and (gradient_norm < self.config.tolerance):
                    self.plot_stick(positions, fracture_vector, iteration, step, max_stress)
                    break

            current_lr = max(current_lr * self.config.decay_factor, self.config.min_learning_rate)

        else:
            self.plot_stick(positions, fracture_vector, iteration, step, max_stress)
            print("Maximum iterations reached without convergence.")
            print(f"Final Gradient Norm: {gradient_norm}")
            print(f"Fracture vector: {fracture_vector}")

        return positions.detach(), energy.detach(), iteration

    def run_simulation(self):
        positions = self.initialize_positions()
        fracture_vector = self.initialize_fracture()
        surface_energy_constants = self.initialize_surface_energy()

        energies = []
        iterations_per_step = []
        positions_over_time = []

        for n in tqdm(range(self.config.time_steps)):
            with torch.no_grad():
                positions[1, 0] = (n) * self.config.d / self.config.time_steps
                positions[1, -1] = -(n) * self.config.d / self.config.time_steps

            position_mask = torch.zeros_like(positions, dtype=torch.bool)
            position_mask[:, 0] = True
            position_mask[:, -1] = True

            optimized_positions, final_energy, iterations = self.optimize_energy(
                positions, fracture_vector, surface_energy_constants, position_mask, n
            )

            energies.append(final_energy.item())
            iterations_per_step.append(iterations)
            if n % self.config.plot_interval == 0 or torch.any(fracture_vector > 0):
                positions_over_time.append(optimized_positions.clone())

        plt.ioff()
        self.plot_results(energies, positions_over_time, iterations_per_step)
        plt.show()
    
    def calculate_ideal_breakage_time(self) -> int:
        minimal_constant = torch.min(self.initialize_surface_energy()).item()
        # we should break whenever the stress is exactly the minimal constant.
        displacement_for_breakage = np.sqrt(minimal_constant / self.config.k_stretch)
        length_per_segment_for_breakage = self.config.L + displacement_for_breakage
        total_length = self.config.N * length_per_segment_for_breakage
        # now use pythagoral to calculate the y-displacement
        y_displacement = np.sqrt(total_length**2 - self.config.W**2)
        breakage_time = y_displacement / (2 * self.config.d)
        return breakage_time

    def plot_results(self, energies, positions_over_time, iterations_per_step):
        time_steps = range(len(energies))

        # (1) Energy over time
        plt.figure()
        plt.plot(time_steps, energies)
        plt.xlabel("Time Step")
        plt.ylabel("Total Energy")
        plt.title("Total Energy Over Time")
        plt.grid(True)
        plt.show()

        # (2) 10 different positions over time
        plt.figure()
        for i in range(10):
            idx = i * (self.config.N // 10)
            y_positions = [pos[1, idx].item() for pos in positions_over_time]
            plt.plot(range(len(positions_over_time)), y_positions, label=f"Position {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Y Position")
        plt.title("Y Positions Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # (3) Number of iterations for each time frame
        plt.figure()
        plt.plot(time_steps, iterations_per_step)
        plt.xlabel("Time Step")
        plt.ylabel("Iterations")
        plt.title("Iterations Per Time Step")
        plt.grid(True)
        plt.show()

def main():
    config = SimulationConfig(time_steps=20)
    simulator = Simulator(config)
    print(f"Ideal breakage time: {simulator.calculate_ideal_breakage_time()}")
    simulator.run_simulation()

if __name__ == "__main__":
    main()
