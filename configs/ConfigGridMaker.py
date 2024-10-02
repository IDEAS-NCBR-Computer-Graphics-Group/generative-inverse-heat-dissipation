import os
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib

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

# Function to compute hash of the solver config
def hash_solver(config):
    # Convert ConfigDict to a regular dictionary
    config_dict = config.to_dict()
    
    # Serialize the dictionary to a string with sorted keys
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Compute the MD5 hash of the string
    return hashlib.md5(config_str.encode()).hexdigest()

# Function to assign a value to a nested ConfigDict
def set_nested_value(config, keys, value):
    nested_config = config
    for key in keys[:-1]:
        if key not in nested_config:
            nested_config[key] = ml_collections.ConfigDict()
        nested_config = nested_config[key]
    nested_config[keys[-1]] = value
 
def evaluate_config_file_name(savedir, params):
    config = ml_collections.ConfigDict()
    # Assign the parameters to the nested structure in config
    for param_key, param_value in params.items():
        keys = param_key.split('.')
        set_nested_value(config, keys, param_value)
    config_hash = hash_solver(config)  # .to_dict()
    
    # Define file path
    config_filename = os.path.join(save_dir, f'config_{config_hash}.py')
    return config_filename


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
    print(f"Saved config {i} to {config_filename}")
