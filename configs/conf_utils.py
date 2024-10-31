import os
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib
import numpy as np



def exp_schedule(min_value, max_value, n, dtype=float):
    return np.exp(np.linspace(np.log(min_value), np.log(max_value), n)).astype(dtype)

def lin_schedule(min_value, max_value, n, dtype=float):
    return np.linspace(min_value, max_value, n).astype(dtype)

def cosine_beta_schedule(min_value, max_value, n, s=0.008, dtype=float):
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
    
    return betas_scaled.astype(dtype), alphas_scaled.astype(dtype) 

def inv_cosine_aplha_schedule(min_value, max_value, n, s=0.008, dtype=float):
    """
    Insipredd by schedule proposed in https://arxiv.org/abs/2102.09672
    """
    x = np.linspace(0, n, n)
    alphas_cumprod = np.cos(((x / n) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Rescale 1-alphas_cumprod
    alphas_inv_scaled =  (alphas_cumprod) * (max_value - min_value) + min_value
    return np.flip(alphas_inv_scaled).astype(dtype) 

def tanh_schedule(min_value, max_value, n, steepness = 0.005, dtype=float):
    x = np.linspace(-500, 500, n)
    result = (np.tanh(steepness*x) + 1) / 2
    result_scaled = result * (max_value - min_value) + min_value
    return result_scaled.astype(dtype)

def log_schedule(min_value, max_value, n, log_base=2.0, dtype=int):
    starting_lbm_steps_pow = np.emath.logn(log_base, min_value)
    final_lbm_steps_pow = np.emath.logn(log_base, max_value)
    
    # python 3.10 --> math.pow 
    # python 3.12 --> np.pow
    if np.pow(log_base, final_lbm_steps_pow) != max_value:
        final_lbm_steps_pow += 2 * np.finfo(float).eps
        
    schedule = np.logspace(
        starting_lbm_steps_pow,final_lbm_steps_pow,
        n, base=log_base)
    
    return schedule.astype(dtype)

def hash_int(int_value):
    return hashlib.md5(json.dumps(int_value).encode()).hexdigest()

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
        elif callable(obj):  # Check if the object is a function
            return None  # Or any other placeholder you prefer
        else:
            return obj

    config_dict = convert_numpy_to_list(config_dict)
    
    # Serialize the dictionary to a string with sorted keys
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Compute the MD5 hash of the string
    return hashlib.md5(config_str.encode()).hexdigest()


def hash_joiner(hash_list, num_bits=32):
    """
    Joins a list of hashes into a single hash of variable bit length using hashlib.

    Args:
        hash_list (list): A list of hash strings.
        num_bits (int): The number of bits to use for the hash (must be a multiple of 4).

    Returns:
        str: The combined hash as a hexadecimal string with the length determined by num_bits.
    """
    if num_bits % 4 != 0:
        raise ValueError("Number of bits must be a multiple of 4")

    # Use a hashlib hashing function (e.g., MD5 or SHA256)
    combined_hash = hashlib.md5()
    for hash_str in hash_list:
        combined_hash.update(hash_str.encode())

    # Get the full hash as an integer
    full_hash_int = int(combined_hash.hexdigest(), 16)

    # Mask the hash to the desired number of bits
    max_value = (1 << num_bits) - 1
    shortened_hash = full_hash_int & max_value

    # Calculate the number of hex digits needed to represent the bit length
    num_hex_digits = num_bits // 4
    return f"{shortened_hash:0{num_hex_digits}x}"

# def hash_joiner(hash_list):
#     """
#     Joins a list of hashes into a single, short hash.
#
#     Args:
#         hash_list (list): A list of hash strings.
#
#     Returns:
#         str: The combined hash string.
#     """
#     combined_hash = hashlib.md5()
#     for hash_str in hash_list:
#         combined_hash.update(hash_str.encode())
#     return combined_hash.hexdigest()

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
    config_hash = hash_joiner([hash_solver(config)], num_bits=32)  # .to_dict()
    
    # TODO: First, the default config shall be read from file. Then update values of specific keys.
    # config_filename = os.path.join(savedir, f"config_{config.stamp.fwd_solver_hash}_{config.stamp.model_optim_hash}")
    # Define file path
    config_filename = os.path.join(savedir, f'config_{config_hash}.py')
    return config_filename