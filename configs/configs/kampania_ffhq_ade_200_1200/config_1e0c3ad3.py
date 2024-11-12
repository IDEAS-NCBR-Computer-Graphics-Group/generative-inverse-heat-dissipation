
import ml_collections
import numpy as np
from numpy import array
import torch
from configs import conf_utils
from configs.configs.kampania_ffhq_ade_200_1200 import default_lbm_ffhq_128_config as default_config

def get_config():
    config = default_config.get_default_configs()
            
    config.optim.lr = 2e-05
    config.solver.cs2 = 0.3333333333333333
    config.training.batch_size = 8
    config.training.n_iters = 500001
    config.turbulence.turb_intensity =  conf_utils.lin_schedule(0.0001, 0.0001, 1200, dtype=np.float32)

    stamp = config.stamp
    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    # config
    config.solver.corrupt_sched = conf_utils.exp_schedule(1, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)

    are_steps_unique = True
    if are_steps_unique:
        config.solver.corrupt_sched = np.unique(config.solver.corrupt_sched)
        config.solver.max_fwd_steps = len(config.solver.corrupt_sched)
        config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    
    return config
    