import os, shutil
import re
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib

from configs.conf_utils import evaluate_config_file_name

def main():
    # Define the hyperparameter grid
    # param_grid = {
    #     'training.batch_size': [16, 32, 64],
    #     'training.n_iters': [10001],
    #     'optim.lr': [1e-4, 5e-5, 2e-5, 1e-5],
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    # }
    
    param_grid = {
        'training.batch_size': [64],
        'training.n_iters': [1001],
        'optim.lr': [2e-5, 1e-5],
        'turbulence.turb_intensity' : [0, 1E-4],
        'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    }

    # Create the grid
    grid = ParameterGrid(param_grid)
    # Define the directory to save the config files
    save_dir =os.path.join("configs","campaign_ffhq_ns_128")
    os.makedirs(save_dir, exist_ok=True)
    # Define the default config
    default_cfg_dir_list =  ["configs", "ffhq", "res_128"]
    default_cfg_file = "default_lbm_ffhq_128_config.py"

    shutil.copy(os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file),
                save_dir)

    # Extract the module path and file name without extension
    module_path = os.path.join(*default_cfg_dir_list).replace(os.sep, '.')
    file_name = re.match(r"(.*)\.py", default_cfg_file).group(1)

    import_path = os.path.join(save_dir).replace(os.sep, '.')
    # Construct the default_cfg_str
    default_cfg_str = f"from {import_path} import {file_name} as default_config"

    # Iterate over each configuration in the grid and save it

    for i, params in enumerate(grid):
        config_filename = evaluate_config_file_name(save_dir, params)
        print(f"writing case {i}/{len(grid)} to {config_filename}")
        # Save the updated configuration by modifying only the necessary fields
        with open(config_filename, 'w') as f:
            f.write(f"""
import ml_collections
import numpy as np
import torch
from configs import conf_utils
{default_cfg_str}

def get_config():
    config = default_config.get_default_configs()
            """)
                
            f.write(f"\n") # flush 
            
            # Write only the parameters that were changed
            for param_key, param_value in params.items():
                f.write(f"    config.{param_key} = {repr(param_value)}\n")

            f.write(f"""
    stamp = config.stamp

    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    """)

            f.write(
    """
    return config
    """)

if __name__ == '__main__':
    main()
