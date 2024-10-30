import os, shutil
import re
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib
import sys
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

from configs.conf_utils import evaluate_config_file_name

# def main():
#     # Define the save directory and make sure it exists
#     save_dir = os.path.join("configs", "campaign_ffhq_ns_128")
#     os.makedirs(save_dir, exist_ok=True)

#     # Define the default configuration location
#     default_cfg_dir_list = ["configs", "ffhq", "res_128", "ade"]
#     default_cfg_file = "default_lbm_ffhq_128_config.py"

#     # Debugging path
#     src_path = os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file)
#     print(f"Source path: {src_path}")

#     # Attempt to copy the file
#     src_path = "/raid/NFS_SHARE/home/jakub.meixner/generative-inverse-heat-dissipation/configs/ffhq/res_128/ade/default_lbm_ffhq_128_config.py"
#     shutil.copy(src_path, save_dir)


#     src_path = os.path.join("configs", "ffhq", "res_128", "ade", default_cfg_file)
#     print(f"Absolute source path: {os.path.abspath(src_path)}")

def main():
    # Define the hyperparameter grid
    save_dir = os.path.join("configs","campaign_ffhq_ade_128")
    os.makedirs(save_dir, exist_ok=False)
    param_grid = {
        'training.batch_size': [32],
        'training.n_iters': [50001],
        'optim.lr': [1e-4, 5e-5, 2e-5, 1e-5],
        'turbulence.turb_intensity' : [0, 1E-4],
        'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    }
    
    # param_grid = {
    #     'training.batch_size': [64],
    #     'training.n_iters': [1001],
    #     'optim.lr': [2e-5, 1e-5],
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    # }

    default_cfg_dir_list = ["ffhq", "res_128", "ade"]
    default_cfg_file = "default_lbm_ffhq_128_config.py"
    src_path = os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file)
    print(f"Source path: {src_path}")
    
    
    shutil.copy(src_path, save_dir)

    # Create the grid
    grid = ParameterGrid(param_grid)
    # Define the directory to save the config files
    # Define the default config

    shutil.copy(os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file),
                save_dir)

    # Extract the module path and file name without extension
    module_path = os.path.join(*default_cfg_dir_list).replace(os.sep, '.')
    file_name = re.match(r"(.*)\.py", default_cfg_file).group(1)

    import_path = os.path.join( os.path.join("configs", save_dir)).replace(os.sep, '.')
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
    config.solver.lin_sched = True


    # config.solver.n_denoising_steps = 50
    # config.solver.max_fwd_steps = solver.n_denoising_steps + 1 # corruption_amount = np.random.randint(self.min_steps, self.max_steps) thus we need to add +1 as max_fwd_steps is excluded from tossing
    config.solver.final_lbm_step = 50
    if config.solver.lin_sched: 
        config.solver.corrupt_sched = np.linspace(
            config.solver.min_fwd_steps, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)
    else:
        config.solver.lbm_steps_base = 2.0
        config.solver.starting_lbm_steps_pow = np.emath.logn(config.solver.lbm_steps_base, config.solver.min_fwd_steps)
        config.solver.final_lbm_steps_pow = np.emath.logn(config.solver.lbm_steps_base, config.solver.final_lbm_step)
        if np.math.pow(config.solver.lbm_steps_base, config.solver.final_lbm_steps_pow) != config.solver.final_lbm_step:
            config.solver.final_lbm_steps_pow += 2*np.finfo(float).eps
        config.solver.corrupt_sched = np.logspace(
            config.solver.starting_lbm_steps_pow, config.solver.final_lbm_steps_pow,
            config.solver.max_fwd_steps, base=config.solver.lbm_steps_base, dtype=int)
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    """)

            f.write(
    """
    return config
    """)

if __name__ == '__main__':
    main()
