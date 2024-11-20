import ml_collections
import numpy as np
from numpy import array
import torch
from configs import conf_utils
from configs.configs.kampania_ffhq_ade_200_1199 import default_lbm_ffhq_128_config as default_config

def get_config():
    config = default_config.get_default_configs()
            
    config.optim.lr = 2e-05
    config.solver.cs2 = 0.3333333333333333
    config.training.batch_size = 8
    config.training.n_iters = 200001
    config.turbulence.turb_intensity = conf_utils.lin_schedule(0, 0, 1199, dtype=np.float32)

    stamp = config.stamp

    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    # config.solver.final_lbm_step = 1199
                            

    config.solver.corrupt_sched = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
    183, 184, 185, 186, 187, 189, 191, 193, 196, 199, 203, 208,
    214, 221, 229, 239, 251
])
    # config.solver.corrupt_sched = conf_utils.exp_schedule(1, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)

    # niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    # niu_sched = conf_utils.lin_schedule(1 / 6,  1 / 6, solver.final_lbm_step, dtype=np.float32)
    # niu_sched = conf_utils.tanh_schedule(1E-4 *1./ 6,  1./ 6, solver.max_fwd_steps, dtype=np.float32)
    niu_sched = [
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05,
        1.71302746e-05, 2.10968001e-05, 2.59817769e-05, 3.19978730e-05,
        3.94069997e-05, 4.85317143e-05, 5.97692620e-05, 7.36088707e-05,
        9.06530492e-05, 1.11643817e-04, 1.37495011e-04, 1.69332063e-04,
        2.08541005e-04, 2.56828800e-04, 3.16297664e-04, 3.89536579e-04,
        4.79734008e-04, 5.90816708e-04, 7.27620673e-04, 8.96101678e-04,
        1.10359456e-03, 1.35913255e-03, 1.67384051e-03, 2.06141929e-03,
        2.53874217e-03, 3.12658945e-03, 3.85055313e-03, 4.74215103e-03,
        5.84019896e-03, 7.19250055e-03, 8.85792839e-03, 1.09089871e-02,
        1.34349696e-02, 1.65458449e-02, 2.03770453e-02, 2.50953624e-02,
        3.09062087e-02, 3.80625600e-02, 4.68759687e-02, 5.77301275e-02,
        7.10975734e-02, 8.75602594e-02, 1.07834890e-01, 1.32804124e-01,
        1.63554999e-01, 1.00713129e-01, 1.24033315e-01, 1.52753304e-01,
        1.25415616e-01, 1.54455677e-01, 1.42664987e-01, 1.40559320e-01,
        1.44254930e-01, 1.52277639e-01, 1.64095416e-01
    ]    
    config.solver.niu = config.solver.bulk_visc = np.array(niu_sched)
    config.solver.hash = conf_utils.hash_solver(config.solver)


    are_steps_unique = True
    if are_steps_unique:
        config.solver.corrupt_sched = np.unique(config.solver.corrupt_sched)
        config.solver.max_fwd_steps = len(config.solver.corrupt_sched)
        config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    

    return config
    