import ml_collections
from configs.ffhq import default_lbm_ffhq_ade_128_config as default_lbm_ffhq_config
import numpy as np
import torch
from configs import conf_utils

def get_config():
    return get_default_configs()

def get_default_configs():
    config = default_lbm_ffhq_config.get_default_configs()

    turbulence = config.turbulence
    solver = config.solver
    # niu_sched  = conf_utils.exp_schedule(1E-4 * 1./6., 1./6., n)
    niu_sched  = conf_utils.lin_schedule(0.5 * 1/6, 0.5 * 1/6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)
    
    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    
    return config
    