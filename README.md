[![Run simulations and create videos](https://github.com/duesenfranz/quasistatic_evolutions_simulations/actions/workflows/run-simulations.yml/badge.svg)](https://github.com/duesenfranz/quasistatic_evolutions_simulations/actions/workflows/run-simulations.yml)

# Simulations for Quasi-Static Evolutions

An elastic rod is simulated as $n$ connected nodes.
The two outmost nodes are controlled, while all other nodes move as dictated by a (discrete) quasistatic evolution.
More concretely, we let the two outermost nodes evolve for some time small time $\delta$, after which we let the system
transition to a critical point via gradient descent.
After the system has reached a critical point, we iterate the procedure for the whole time horizon.

## Simulation Results

<table>
    <thead>
        <tr>
            <th>$\delta = \frac{1}{15}$</th>
            <th>$\delta = \frac{1}{240}$</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> 
                https://github.com/user-attachments/assets/20e84524-261e-4c28-a182-f090c5de341e
            </td>
            <td>  
                https://github.com/user-attachments/assets/0183c5fa-589b-45df-893c-678b892c335f
            </td>  
        </tr>
        <tr>
            <td colspan=2><img src="https://github.com/duesenfranz/quasistatic_evolutions_simulations/releases/download/rolling/energy_sums.png" width="100%"/></td>
        </tr>
        <tr>
            <td colspan=2><img src="https://github.com/duesenfranz/quasistatic_evolutions_simulations/releases/download/rolling/combined.png" width="100%"/></td>
        </tr>
    </tbody>
</table>

## Running the simulations

To run the simulations, you need to make sure you have `git`, `python3` and `poetry` installed and then need to follow these steps:

1. Clone the repository with `git clone git@github.com:duesenfranz/quasistatic_evolutions_simulations.git && cd quasistatic_evolutions_simulations`
2. Install the package and the dependencies with `poetry install`
3. Run the simulations with `poetry run python quasistatic_evol/rod_simulator.py`
4. Create the plots with `poetry run python quasistatic_evol/plotting.py`
5. Create the videos with `poetry run python quasistatic_evol/create_videos.py`

### Adapting the simulations

You can change all configurations in `quasistatic_evol/configuration.json`. You can see all possible configuration options in [SimulatorConfig](https://github.com/duesenfranz/quasistatic_evolutions_simulations/blob/main/quasistatic_evol/rod_simulator.py#L22).
**Warning**: The total number of frames in the video needs to be a multiple of all the step numbers.
