import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import numpy as np

# Constants
H = 1.0
W = 10.0
N = 30
C_surface_average = 0.0006
C_surface_std = 0.0001
L = W / N
k_stretch = 1.0
c_angle = 0.0
learning_rate = 0.2
decay_factor = 0.99
tolerance = 5e-6
max_iterations = 100000
d = 2.5
time_steps = 50
plot_interval = N // 10
error_norm_p = 4

# Global variables for the figure and axes
fig = None
ax = None

# Enable interactive mode
plt.ion()


def initialize_surface_energy() -> Tensor:
    """
    Initializes the surface energy for each fracture.

    :return: An (N-1) tensor of surface energies.
    """
    return torch.normal(C_surface_average, C_surface_std, (N - 1,))


def initialize_positions() -> Tensor:
    """
    Initializes the 2D positions of the stick as an (N, 2) array.

    :return: A (2, N) tensor with x and y positions for the stick.
    """
    x_positions = torch.linspace(0, W, N)
    y_positions = torch.zeros(N)
    return torch.stack([x_positions, y_positions])


def initialize_fracture() -> Tensor:
    """
    Initializes the fracture vector. Initially, there are no fractures.
    """
    return torch.zeros(N - 1)


def compute_angles_from_positions(positions: Tensor) -> Tensor:
    """
    Computes the angles between consecutive segments of the stick.

    :param positions: A (2, N) tensor of positions.
    :return: An (N-1) tensor of angles between consecutive segments.
    """
    x_positions = positions[0]
    y_positions = positions[1]
    dx = x_positions[1:] - x_positions[:-1]
    dy = y_positions[1:] - y_positions[:-1]
    return torch.atan2(dy, dx)


def compute_surface_energy(
    fracture_vector: Tensor, surface_energy_constants: Tensor
) -> Tensor:
    """
    Computes the surface energy based on the fracture vector.

    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :param surface_energy_constants: An (N-1) tensor of surface energy constants.
    :return: The total surface energy.
    """
    return torch.sum(fracture_vector * surface_energy_constants)


def compute_stress_vector(positions: Tensor, fracture_vector: Tensor) -> Tensor:
    """
    Computes the stress vector based on the positions of the stick segments and fracture state.

    :param positions: A (2, N) tensor of positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: An (N-1) tensor of stress values for each segment, adjusted for fractures.
    """
    x_positions = positions[0]
    y_positions = positions[1]

    # Compute dx, dy, and distances for all segments
    dx = x_positions[1:] - x_positions[:-1]
    dy = y_positions[1:] - y_positions[:-1]
    distances = torch.sqrt(dx**2 + dy**2)

    # Compute stress for all segments
    stress = k_stretch * (distances - L)

    # Apply fracture weights
    weights = 1 - fracture_vector
    adjusted_stress = stress * weights

    return adjusted_stress


def compute_elastic_energy(positions: Tensor, fracture_vector: Tensor) -> Tensor:
    """
    Computes the elastic energy based on stress and fracture state.

    :param positions: A (2, N) tensor of positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: The total elastic energy.
    """
    stress = compute_stress_vector(positions, fracture_vector)
    
    # Compute elastic energy
    elastic_energy = stress**2

    total_energy = torch.sum(elastic_energy)

    return total_energy


def total_energy(
    positions: Tensor, fracture_vector: Tensor, surface_energy_constants: Tensor
) -> Tensor:
    """
    Computes the total energy (surface + elastic) of the system.

    :param positions: A (2, N) tensor of positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :param surface_energy_constants: An (N-1) tensor of surface energy constants.
    :return: The total energy of the system.
    """
    surface_energy = compute_surface_energy(fracture_vector, surface_energy_constants)
    elastic_energy = compute_elastic_energy(positions, fracture_vector)
    return surface_energy + elastic_energy


def plot_stick(positions: Tensor, fracture_vector: Tensor, iteration: int, step: int):
    """
    Plots the stick's current positions in 2D, coloring each segment based on stress.

    :param positions: A (2, N) tensor of current positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :param iteration: Current iteration number.
    :param step: Current time step.
    """
    global fig, ax
    
    x_positions = positions[0].detach().cpu().numpy()
    y_positions = positions[1].detach().cpu().numpy()

    # Compute stress vector
    stress = compute_stress_vector(positions, fracture_vector).detach().cpu().numpy()

    if fig is None or ax is None:
        # Create a new figure and axis if they don't exist
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        # Clear the existing plot
        ax.clear()
    
    # Create a colormap
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=0, vmax=0.5)

    # Plot segments with color based on stress
    # for i in range(len(x_positions) - 1):
    #     ax.plot([x_positions[i], x_positions[i+1]], 
    #             [y_positions[i], y_positions[i+1]], 
    #             color=cmap(norm(stress[i])),
    #             linewidth=2)

    # Plot points
    ax.plot(x_positions, y_positions, 'ko', markersize=4)

    ax.set_title(f"Stick Configuration at Step {step}, Iteration {iteration}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    # Set the y-axis limits to always range from -d to d
    ax.set_ylim(-d, d)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if len(ax.figure.axes) > 1:
        # If colorbar exists, update it
        ax.figure.axes[1].remove()
    plt.colorbar(sm, ax=ax, label='Stress')

    # This allows the plot to update without blocking
    plt.draw()
    plt.pause(0.2)  # Pause briefly to allow the figure to update


def optimize_energy(
    positions: Tensor,
    fracture_vector: Tensor,
    surface_energy_constants: Tensor,
    position_mask: Tensor,
    step: int,
) -> tuple[Tensor, Tensor, int]:
    """
    Performs manual gradient descent to minimize the total energy.

    :param positions: A (2, N) tensor of initial positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :param surface_energy_constants: An (N-1) tensor of surface energy constants.
    :param position_mask: A (2, N) tensor of booleans indicating fixed positions.
    :param step: Current time step.
    :return: A tuple of the optimized positions, the final minimized energy, and iterations.
    """
    total_energy_fct = torch.jit.trace(
        total_energy, (positions, fracture_vector, surface_energy_constants)
    )
    positions.requires_grad_(True)
    fracture_vector.requires_grad_(True)

    for iteration in tqdm(range(max_iterations), leave=False):

        energy = total_energy_fct(positions, fracture_vector, surface_energy_constants)
        energy.backward()

        with torch.no_grad():
            positions.grad[position_mask] = 0  # Fix masked positions
            positions -= learning_rate * positions.grad
            fracture_vector -= learning_rate * fracture_vector.grad * decay_factor ** iteration

            # Ensure fracture_vector stays between 0 and 1
            fracture_vector.clamp_(0, 1)

            # If the fracture vector is close to 0 or 1, we must ignore the corresponding gradient
            # for the convergence check.
            fractor_gradient_finished = (fracture_vector == 0) | (fracture_vector == 1)
            grad_norm = torch.norm(positions.grad)
            gradient_norm = np.linalg.norm(
                positions.grad.cpu().numpy().flatten(), ord=error_norm_p
            )

            if fractor_gradient_finished.all() and (gradient_norm < tolerance):
                # print(f"Converged after {iteration} iterations.")
                # print(f"breaks: {fracture_vector.sum()}")
                plot_stick(positions, fracture_vector, iteration, step)
                positions.grad.zero_()
                fracture_vector.grad.zero_()
                break
            positions.grad.zero_()
            fracture_vector.grad.zero_()
    else:
        print("Maximum iterations reached without convergence.")
        print(f"Final Gradient Norm: {grad_norm}")

    return positions.detach(), energy.detach(), iteration


def main():
    positions = initialize_positions()
    fracture_vector = initialize_fracture()
    surface_energy_constants = initialize_surface_energy()

    energies = []
    iterations_per_step = []
    positions_over_time = []

    for n in tqdm(range(time_steps)):
        # Update boundary conditions
        with torch.no_grad():
            positions[1, 0] = (n) * d / time_steps
            positions[1, -1] = -(n) * d / time_steps

        # Fix leftmost and rightmost points (x and y positions)
        position_mask = torch.zeros_like(positions, dtype=torch.bool)
        position_mask[:, 0] = True
        position_mask[:, -1] = True

        # Optimize energy
        optimized_positions, final_energy, iterations = optimize_energy(
            positions, fracture_vector, surface_energy_constants, position_mask, n
        )

        # Save data for plotting
        energies.append(final_energy.item())
        iterations_per_step.append(iterations)
        if n % plot_interval == 0 or torch.any(fracture_vector > 0):
            positions_over_time.append(optimized_positions.clone())

        # print(f"Iteration {n}: Energy = {final_energy.item()}")

    # Plot results
    plt.ioff()
    plot_results(energies, positions_over_time, iterations_per_step)

    # Add this at the end of the main function
    plt.show()  # This will keep the final plot open


def plot_results(energies, positions_over_time, iterations_per_step):
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
        idx = i * (N // 10)
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


if __name__ == "__main__":
    main()
