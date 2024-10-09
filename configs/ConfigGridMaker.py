import os
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib

from configs.conf_utils import evaluate_config_file_name


# Define the hyperparameter grid
param_grid = {
    'solver.fiuu': [0.001, 0.01, 0.1],
    'solver.boo': [16, 32, 64],
    'solver.bzdziu': [1/6, 1/12, 1/24],
    'training.n_iters': [1001, 10001, 20001]
}

# Create the grid
grid = ParameterGrid(param_grid)

# Define the directory to save the config files
save_dir = 'test_configs'
os.makedirs(save_dir, exist_ok=True)


# Iterate over each configuration in the grid and save it
for i, params in enumerate(grid):
    config_filename = evaluate_config_file_name(save_dir, params)

    # Save the updated configuration by modifying only the necessary fields
    with open(config_filename, 'w') as f:
        f.write("""
from configs.mnist import default_mnist_configs
import ml_collections
import numpy as np
import torch

    def get_config():
        config = default_mnist_configs.get_default_configs()
        """)
            
            f.write(f"\n") # flush 
            
            # Write only the parameters that were changed
            for param_key, param_value in params.items():
                f.write(f"    config.{param_key} = {repr(param_value)}\n")

            f.write("    return config\n")

def main():
    # Define the hyperparameter grid
    param_grid = {
        'solver.fiuu': [0.001, 0.01, 0.1],
        'solver.boo': [16, 32, 64],
        'solver.bzdziu': [1/6, 1/12, 1/24],
        'training.n_iters': [1001, 10001, 20001]
    }

    # Create the grid
    grid = ParameterGrid(param_grid)

    # Define the directory to save the config files
    save_dir = 'test_configs'
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each configuration in the grid and save it
    for i, params in enumerate(grid):
        config_filename = evaluate_config_file_name(save_dir, params)
        save_config(config_filename, params)

        print(f"Saved config {i} to {config_filename}")

if __name__  == '__main__':
    main()