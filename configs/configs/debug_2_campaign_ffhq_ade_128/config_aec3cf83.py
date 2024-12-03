
import ml_collections
import numpy as np
import torch
from configs import conf_utils
from configs.configs.debug_2_campaign_ffhq_ade_128 import default_lbm_ade_ffhq_128_config as default_config

def get_config():
    config = default_config.get_default_configs()
    solver = config.solver
    turbulence = config.turbulence
    
    
    ### GRID MAKER ###
            
    config.solver.n_denoising_steps = 200
    config.solver.are_steps_unique = False

    ### GRID MAKER DONE ###
    
    
    if config.solver.are_steps_unique:
        config.solver.corrupt_sched = np.unique(config.solver.corrupt_sched)
        config.solver.max_fwd_steps = len(config.solver.corrupt_sched)
        config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
        
    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, config.solver.final_lbm_step)
    # niu_sched = conf_utils.lin_schedule(1E-4*1/6, 1./6, config.solver.final_lbm_step)
    # config.solver.niu = config.solver.bulk_visc = niu_sched
    
    stamp = config.stamp

    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    
    return config
    