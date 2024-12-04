import os, shutil
import re
import ml_collections
from sklearn.model_selection import ParameterGrid
import json
import hashlib
import numpy as np
import sys

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
import conf_utils
from conf_utils import evaluate_config_file_name, lin_schedule

def main():
    save_dir = os.path.join("configs","debug_recalculation_camp_3")
    os.makedirs(save_dir, exist_ok=False)
    param_grid = {
        'training.batch_size': [8],
        'training.n_iters': [200001],
        'optim.lr': [2e-5],
        # 'solver.cs2' : [1./3.],
        'solver.Pe' : [1e-1],
        'model.K' : [200],
        'model.blur_schedule' : [np.array([0] + list(np.exp(np.linspace(np.log(0.5), np.log(32), 200)) ))]
    }

    default_cfg_dir_list = ["ffhq"]
    default_cfg_file = "default_lbm_ade_ffhq_128_config.py"
    src_path = os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file)
    print(f"Source path: {src_path}")
    
    
    shutil.copy(src_path, save_dir)

    grid = ParameterGrid(param_grid)
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
        with open(config_filename, 'w') as f:
            f.write(f"""
import ml_collections
import numpy as np
from numpy import array
import torch
from configs import conf_utils
from configs.match_sim_numbers import get_ihd_solver_setup, u_from_Pe

{default_cfg_str}

def get_config():
    config = default_config.get_default_configs()
            """)                                        
            f.write(f"\n") # flush 

            # Write only the parameters that were changed
            for param_key, param_value in params.items():
                if isinstance(param_value, str) and param_value.startswith(('conf_utils','config.')):
                    f.write(f"    config.{param_key} = {param_value}\n")
                else:
                    f.write(f"    config.{param_key} = {repr(param_value)}\n")                                    
                                    
            f.write(f"""
    stamp = config.stamp

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, config.solver.final_lbm_step)
    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)

    config = get_ihd_solver_setup(config) # get IHD parammeters and recompute to LBM 
    config.solver.bulk_visc = config.solver.niu

    config.turbulence.turb_intensity = u_from_Pe(config.solver.Pe, config.solver.niu, config.data.image_size)

    config.solver.hash = conf_utils.hash_solver(config.solver)



    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    """)
            f.write(
    """

    return config
    """)


if __name__ == '__main__':
    main()
