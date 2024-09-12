import torch
from torch import Tensor
from tqdm import tqdm

# Constants
H = 1.0
W = 10.0
N = 100
C_surface = 0.5
L = W / N
k_stretch = 1.0
learning_rate = 0.01
tolerance = 1e-3
max_iterations = 1000


def initialize_positions() -> Tensor:
    """
    Initializes the 2D positions of the stick as an (N, 2) array.

    :return: A (2, N) tensor with x and y positions for the stick.
    """
    x_positions = torch.linspace(0, W, N)
    y_positions = torch.ones(N) * H
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


def compute_surface_energy(fracture_vector: Tensor) -> Tensor:
    """
    Computes the surface energy based on the fracture vector.

    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: The total surface energy.
    """
    return C_surface * torch.sum(fracture_vector)


def compute_elastic_energy(positions: Tensor, fracture_vector: Tensor) -> Tensor:
    """
    Computes the elastic energy based on angle differences and stretching/compression.

    :param positions: A (2, N) tensor of positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: The total elastic energy.
    """
    angles = compute_angles_from_positions(positions)
    x_positions = positions[0]
    y_positions = positions[1]
    total_energy = 0.0

    for i in range(N - 2):
        angle_diff = torch.abs(angles[i + 1] - angles[i])
        weight = 1 - fracture_vector[i]
        total_energy += weight * angle_diff

        dx = x_positions[i + 1] - x_positions[i]
        dy = y_positions[i + 1] - y_positions[i]
        distance = torch.sqrt(dx**2 + dy**2)
        stretch_energy = k_stretch * (distance - L) ** 2
        total_energy += stretch_energy * weight

    return total_energy


def total_energy(positions: Tensor, fracture_vector: Tensor) -> Tensor:
    """
    Computes the total energy (surface + elastic) of the system.

    :param positions: A (2, N) tensor of positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: The total energy of the system.
    """
    surface_energy = compute_surface_energy(fracture_vector)
    elastic_energy = compute_elastic_energy(positions, fracture_vector)
    return surface_energy + elastic_energy


def optimize_energy(
    positions: Tensor, fracture_vector: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Performs manual gradient descent to minimize the total energy.

    :param positions: A (2, N) tensor of initial positions.
    :param fracture_vector: An (N-1) tensor of floats between 0 and 1.
    :return: A tuple of the optimized positions and the final minimized energy.
    """
    positions.requires_grad_(True)
    fracture_vector.requires_grad_(True)

    for iteration in tqdm(range(max_iterations)):
        energy = total_energy(positions, fracture_vector)
        energy.backward()

        with torch.no_grad():
            positions -= learning_rate * positions.grad
            fracture_vector -= learning_rate * fracture_vector.grad

            # Ensure fracture_vector stays between 0 and 1
            fracture_vector.clamp_(0, 1)

            # If the fracture vector is close to 0 or 1, we must ignore the corresponding gradient
            # for the convergence check.
            fractor_gradient_finished = (fracture_vector == 0) | (fracture_vector == 1)
            grad_norm = torch.norm(positions.grad)

            positions.grad.zero_()
            fracture_vector.grad.zero_()

            if grad_norm < tolerance and fractor_gradient_finished.all():
                print(f"Converged after {iteration} iterations.")
                break
    else:
        print("Maximum iterations reached without convergence.")
        print(f"Final Gradient Norm: {grad_norm}")

    return positions.detach(), energy.detach()


if __name__ == "__main__":
    positions = initialize_positions()
    fracture_vector = initialize_fracture()

    optimized_positions, final_energy = optimize_energy(positions, fracture_vector)

    print("Final Optimized Energy:", final_energy.item())
    print("Optimized Stick Configuration:\n", optimized_positions)
    print("Number of Fractures:", torch.sum(fracture_vector).item())
