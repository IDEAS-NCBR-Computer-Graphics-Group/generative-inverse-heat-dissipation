import os, shutil
import re
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib

import conf_utils
from conf_utils import evaluate_config_file_name

def main():
    # Define the hyperparameter grid
    
    # third campaign
    param_grid = {
        # 'solver.final_lbm_step' : [500],
        'solver.n_denoising_steps' : [200] ,
        # 'solver.max_fwd_steps': ['config.solver.n_denoising_steps + 1'],
        # 'solver.corrupt_sched' : [
        #     'conf_utils.exp_schedule(config.solver.min_fwd_steps, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)',
        #     'conf_utils.lin_schedule(config.solver.min_fwd_steps, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)'
        #     ],
    
        # 'turbulence.turb_intensity' : [
        #     'conf_utils.lin_schedule(0., 0., config.solver.final_lbm_step, dtype=np.float32)',
        #     'conf_utils.exp_schedule(1E-6, 1E-3, config.solver.final_lbm_step, dtype=np.float32)',
        #     'conf_utils.lin_schedule(1E-6, 5E-4, config.solver.final_lbm_step, dtype=np.float32)'
        #     ],
        'solver.are_steps_unique': [True, False] 

    }
    
    # second campaign
    # param_grid = {
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.are_steps_unique': [True, False] 
    # }
    
    # first campaign
    # param_grid = {
    #     'training.batch_size': [16, 32, 64],
    #     'training.n_iters': [20001],
    #     'optim.lr': [1e-4, 5e-5, 2e-5, 1e-5],
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    # }
    
    # param_grid = {
    #     'training.batch_size': [64],
    #     'training.n_iters': [1001],
    #     'optim.lr': [2e-5, 1e-5],
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    # }  # solver.corrupt_sched = conf_utils.lin_schedul(
    #         solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
    

    # Create the grid
    grid = ParameterGrid(param_grid)
    # Define the directory to save the config files
    name = "debug_2_campaign_ffhq_ade_128"

    save_dir =os.path.join("configs",name)
    save_dir_2= os.path.join("configs","configs",name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # Define the default config
    default_cfg_dir_list =  ["ffhq"]
    default_cfg_file = "default_lbm_ade_ffhq_128_config.py"

    shutil.copy(os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file),
                save_dir)
    # Extract the module path and file name without extension
    module_path = os.path.join(*default_cfg_dir_list).replace(os.sep, '.')
    file_name = re.match(r"(.*)\.py", default_cfg_file).group(1)

    import_path = os.path.join(save_dir_2).replace(os.sep, '.')
    
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
    
    
    if config.solver.are_steps_unique:
        config.solver.corrupt_sched = np.unique(config.solver.corrupt_sched)
        config.solver.max_fwd_steps = len(config.solver.corrupt_sched)
        config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
        
    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, config.solver.final_lbm_step)
    # niu_sched = conf_utils.lin_schedule(1E-4*1/6, 1./6, config.solver.final_lbm_step)
    # config.solver.niu = config.solver.bulk_visc = niu_sched  
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
