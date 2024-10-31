
import ml_collections
import numpy as np
import torch
from configs import conf_utils
from configs.test.campaign_ffhq_ade_128 import default_lbm_ffhq_128_config as default_config

def get_config():
    config = default_config.get_default_configs()
            
    config.optim.lr = 0.0001
    config.solver.cs2 = 0.19999999999999998
    config.training.batch_size = 32
    config.training.n_iters = 50001
    config.turbulence.turb_intensity = 0

    stamp = config.stamp

    timesteps_list = [0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        3,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        11,
                        13,
                        15,
                        17,
                        20,
                        23,
                        27,
                        32,
                        37,
                        43,
                        50,
                        59,
                        68,
                        79,
                        92,
                        107,
                        125,
                        145,
                        169,
                        197,
                        229,
                        266,
                        309,
                        359,
                        418,
                        486,
                        565,
                        657,
                        763,
                        887,
                        1032,
                        1199]
    
    timesteps_array = np.array(timesteps_list)
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
        # config.solver.corrupt_sched = np.linspace(
        #     config.solver.min_fwd_steps, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)
        config.solver.corrupt_sched = timesteps_array
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
    
    return config
    