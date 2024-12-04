import ml_collections
from configs.ffhq.res_128.ade import default_lbm_ffhq_128_config as default_lbm_ffhq_config
import numpy as np
import torch
from torchvision import transforms
from configs import conf_utils


def get_config():
    config = default_lbm_ffhq_config.get_config()
    
    turbulence = config.turbulence
    solver = config.solver
    
    niu_sched  = conf_utils.tanh_schedule(1E-4 * 1/6, 1/6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc =  niu_sched
    solver.fwd_solver_hash = conf_utils.hash_solver(solver)

    stamp = config.stamp
    
    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    
    return config