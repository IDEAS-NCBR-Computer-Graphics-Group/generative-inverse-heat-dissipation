
import ml_collections
import numpy as np
from numpy import array
import torch
from configs import conf_utils
from configs.configs.kampania_ffhq_ade_200_1199_sigma_16 import default_lbm_ffhq_128_config as default_config

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
                            
    config.solver.corrupt_sched = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
     63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
     86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
     107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
     125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
     143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
     161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
     179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 193, 195, 198, 201, 205, 210, 215, 222, 230])
     
    # config.solver.corrupt_sched = conf_utils.exp_schedule(1, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)

    # niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    # niu_sched = conf_utils.lin_schedule(1 / 6,  1 / 6, solver.final_lbm_step, dtype=np.float32)
    # niu_sched = conf_utils.tanh_schedule(1E-4 *1./ 6,  1./ 6, solver.max_fwd_steps, dtype=np.float32)
    niu_sched = [1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
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
             1.66666667e-05, 2.03310275e-05, 2.50327217e-05, 3.08217161e-05, 
             3.79494564e-05, 4.67255373e-05, 5.75311490e-05, 7.08356351e-05, 
             8.72168779e-05, 1.07386399e-04, 1.32220265e-04, 1.62797139e-04, 
             2.00445132e-04, 2.46799489e-04, 3.03873621e-04, 3.74146550e-04, 
             4.60670592e-04, 5.67203933e-04, 6.98373864e-04, 8.59877771e-04, 
             1.05873060e-03, 1.30356956e-03, 1.60502926e-03, 1.97620366e-03, 
             2.43321476e-03, 2.99591293e-03, 3.68873903e-03, 4.54178608e-03, 
             5.59210631e-03, 6.88532054e-03, 8.47759973e-03, 1.04381048e-02, 
             1.28519905e-02, 1.58241044e-02, 1.94835407e-02, 2.39892477e-02, 
             2.95369313e-02, 3.63675560e-02, 4.47778112e-02, 5.51329975e-02, 
             6.78828940e-02, 8.35812945e-02, 1.02910061e-01, 1.26708743e-01, 
             1.56011038e-01, 9.60448481e-02, 1.18255900e-01, 1.45603415e-01, 
             1.19516825e-01, 1.47155938e-01, 1.35890094e-01, 1.33852513e-01, 
             1.64806856e-01, 1.44942578e-01, 1.56153880e-01]

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
    