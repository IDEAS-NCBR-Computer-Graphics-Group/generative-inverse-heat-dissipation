import os
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib
import numpy as np

def exp_schedule(min_value, max_value, n):
    return np.exp(np.linspace(np.log(min_value), np.log(max_value), n))

def lin_schedule(min_value, max_value, n):
    return np.linspace(min_value ,max_value, n)

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
    config_filename = os.path.join(savedir, f'config_{config_hash}.py')
    return config_filename