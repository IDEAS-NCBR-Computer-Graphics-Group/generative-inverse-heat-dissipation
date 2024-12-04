import os, shutil
import re
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib

import configs.conf_utils
from configs.conf_utils import evaluate_config_file_name

def main():
    param_grid = {
        'turbulence.turb_intensity' : [
            'conf_utils.lin_schedule(0., 0., config.solver.final_lbm_step, dtype=np.float32)',
            'conf_utils.exp_schedule(1E-6, 1E-2, config.solver.final_lbm_step, dtype=np.float32)',
            'conf_utils.exp_schedule(1E-6, 5E-3, config.solver.final_lbm_step, dtype=np.float32)',
            'conf_utils.exp_schedule(1E-6, 1E-3, config.solver.final_lbm_step, dtype=np.float32)',
            'conf_utils.lin_schedule(1E-6, 5E-4, config.solver.final_lbm_step, dtype=np.float32)',
            'conf_utils.lin_schedule(1E-6, 1E-4, config.solver.final_lbm_step, dtype=np.float32)'
            ],
    }

    grid = ParameterGrid(param_grid)
    save_dir =os.path.join("configs", "mnist", "campaign_small_mnist_ade")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    default_cfg_dir_list =  ["configs", "mnist"]
    default_cfg_file = "default_lbm_ade_small_mnist_config.py"

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
        # Create a dictionary to store the parameters in the desired order
        ordered_params = {}
        for key in param_grid:
            ordered_params[key] = params[key]


        config_filename = evaluate_config_file_name(save_dir, ordered_params)
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
    solver = config.solver
    turbulence = config.turbulence
    
    
    ### GRID MAKER ###
            """)
                
            f.write(f"\n") # flush 
            
            # Write only the parameters that were changed
            for param_key, param_value in ordered_params.items():
                if isinstance(param_value, str) and param_value.startswith(('conf_utils','config.')):
                    f.write(f"    config.{param_key} = {param_value}\n")
                else:
                    f.write(f"    config.{param_key} = {repr(param_value)}\n")

            f.write(f"""
    ### GRID MAKER DONE ###
    
    """)
        
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
