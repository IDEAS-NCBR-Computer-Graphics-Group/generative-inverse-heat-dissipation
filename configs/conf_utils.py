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

def cosine_beta_schedule(n, min_value, max_value, s=0.008):
    """
    Rescaled cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    x = np.linspace(0, n, n)
    alphas_cumprod = np.cos(((x / n) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1-(alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Rescale betas
    betas_scaled = betas * (max_value - min_value) + min_value
    
    # Rescale 1-alphas_cumprod
    alphas_scaled =  alphas_cumprod * (max_value - min_value) + min_value
    
    return betas_scaled, alphas_scaled 

def inv_cosine_aplha_schedule(n, min_value, max_value, s=100):
    """
    Insipredd by schedule proposed in https://arxiv.org/abs/2102.09672
    """
    x = np.linspace(n, 0, n)
    alphas_cumprod = np.cos(((x / n) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Rescale 1-alphas_cumprod
    alphas_inv_scaled =  (alphas_cumprod) * (max_value - min_value) + min_value
    return  alphas_inv_scaled 

def tanh_schedule(min_value, max_value, n, steepness = 0.005):
    x = np.linspace(-500, 500, n)
    result = (np.tanh(steepness*x) + 1) / 2
    result_scaled = result * (max_value - min_value) + min_value
    return result_scaled 

# Function to compute hash of the solver config
def hash_solver(config):
    # Convert ConfigDict to a regular dictionary
    config_dict = config.to_dict()
    
    # Convert NumPy arrays to lists recursively with precision control
    def convert_numpy_to_list(obj, precision=6):  # Added precision parameter
        if isinstance(obj, dict):
            return {k: convert_numpy_to_list(v, precision) for k, v in obj.items()}
        elif isinstance(obj, (np.ndarray, np.generic)):
            return np.round(obj, decimals=precision).tolist()  # Round before converting to list
        else:
            return obj

    config_dict = convert_numpy_to_list(config_dict)
    
    # Serialize the dictionary to a string with sorted keys
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Compute the MD5 hash of the string
    return hashlib.md5(config_str.encode()).hexdigest()

def hash_joiner(hash_list):
    """
    Joins a list of hashes into a single, short hash.

    Args:
        hash_list (list): A list of hash strings.

    Returns:
        str: The combined hash string.
    """
    combined_hash = hashlib.md5()
    for hash_str in hash_list:
        combined_hash.update(hash_str.encode())
    return combined_hash.hexdigest()

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