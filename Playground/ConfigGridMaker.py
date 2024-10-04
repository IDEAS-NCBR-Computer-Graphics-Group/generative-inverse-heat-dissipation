from configs.mnist import default_mnist_configs
import ml_collections
import numpy as np
import torch
import json
import hashlib
from sklearn.model_selection import ParameterGrid

def get_config():
    config = default_mnist_configs.get_default_configs()

    model = config.model
    model.blur_sigma_max = 20
    model.blur_sigma_min = 0.5
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
    model.K = 50
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep

    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.min_init_gray_scale = 0.95
    data.max_init_gray_scale = 1.05
    data.processed_filename = 'lbm_ns_turb_pairs' if config.data.process_pairs else 'lbm_ns_turb'
    data.dataset = 'CORRUPTED_NS_MNIST'

    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100

    solver = config.solver
    solver.type = 'fluid'
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.domain_size = (1.0, 1.0)
    solver.turb_intensity = 1E-4
    solver.noise_limiter = (-1E-3, 1E-3)
    solver.dt_turb = 5 * 1E-4
    solver.k_min = 2.0 * torch.pi / min(solver.domain_size)
    solver.k_max = 2.0 * torch.pi / (min(solver.domain_size) / 1024)
    solver.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    solver.min_steps = 1
    solver.max_steps = 10
    solver.n_denoising_steps = 10

    return config

# Define the hyperparameter grid
param_grid = {
    'turb_intensity': [0.001, 0.01, 0.1],
    'dt_turb': [16, 32, 64],
    'niu': [1/6, 1/12, 1/24]
}

# Create the grid
grid = ParameterGrid(param_grid)

# Function to compute hash of the solver config
def hash_solver(solver):
    solver_str = json.dumps(solver, sort_keys=True)
    return hashlib.md5(solver_str.encode()).hexdigest()

# Save each config in a separate file
def save_config(config, filename):
    with open(filename, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

# Iterate through the grid and create different configs
for params in grid:
    # Get the default configuration
    config = get_config()

    # Update the solver hyperparameters based on the current grid params
    config.solver.turb_intensity = params['turb_intensity']
    config.solver.dt_turb = params['dt_turb']
    config.solver.niu = params['niu']

    # Compute the hash of the solver
    solver_hash = hash_solver(config.solver.to_dict())

    # Create a filename with the hash
    filename = f"{config.data.dataset}_solver_{solver_hash}.json"

    # Save the configuration to a file
    save_config(config, filename)

    print(f"Saved config: {filename}")
