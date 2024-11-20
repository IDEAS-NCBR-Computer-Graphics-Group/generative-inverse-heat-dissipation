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
    save_dir = os.path.join("configs","kampania_ffhq_ade_200_sigma_128")
    os.makedirs(save_dir, exist_ok=False)
    param_grid = {
        'training.batch_size': [8],
        'training.n_iters': [200001],
        'optim.lr': [2e-5],
        'turbulence.turb_intensity' : [conf_utils.lin_schedule(0, 0, 1199, dtype=np.float32)], # , lin_schedule(1e-4, 1e-4, 3100, dtype=np.float32), lin_schedule(1e-3, 1e-3, 3100, dtype=np.float32), lin_schedule(1E-6, 5E-4, 3100, dtype=np.float32)
        'solver.cs2' : [1./3.]
    }

    default_cfg_dir_list = ["ffhq", "res_128", "ade"]
    default_cfg_file = "default_lbm_ffhq_128_config.py"
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

    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    # config.solver.final_lbm_step = 1199

    ####### sigma = 16, L = 128, Fo = 0.0009765625, lbm_iter = 767 #######


    # config.solver.corrupt_sched = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    #  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
    #  86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    #  107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    #  125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    #  143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    #  161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    #  179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 193, 195, 198, 201, 205, 210, 215, 222, 230])

    # niu_sched = [1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 1.66666667e-05, 
    #          1.66666667e-05, 2.03310275e-05, 2.50327217e-05, 3.08217161e-05, 
    #          3.79494564e-05, 4.67255373e-05, 5.75311490e-05, 7.08356351e-05, 
    #          8.72168779e-05, 1.07386399e-04, 1.32220265e-04, 1.62797139e-04, 
    #          2.00445132e-04, 2.46799489e-04, 3.03873621e-04, 3.74146550e-04, 
    #          4.60670592e-04, 5.67203933e-04, 6.98373864e-04, 8.59877771e-04, 
    #          1.05873060e-03, 1.30356956e-03, 1.60502926e-03, 1.97620366e-03, 
    #          2.43321476e-03, 2.99591293e-03, 3.68873903e-03, 4.54178608e-03, 
    #          5.59210631e-03, 6.88532054e-03, 8.47759973e-03, 1.04381048e-02, 
    #          1.28519905e-02, 1.58241044e-02, 1.94835407e-02, 2.39892477e-02, 
    #          2.95369313e-02, 3.63675560e-02, 4.47778112e-02, 5.51329975e-02, 
    #          6.78828940e-02, 8.35812945e-02, 1.02910061e-01, 1.26708743e-01, 
    #          1.56011038e-01, 9.60448481e-02, 1.18255900e-01, 1.45603415e-01, 
    #          1.19516825e-01, 1.47155938e-01, 1.35890094e-01, 1.33852513e-01, 
    #          1.64806856e-01, 1.44942578e-01, 1.56153880e-01]


    ####### sigma = 20, L = 128, Fo = 0.001220703125, lbm_iter = 1199 #######


    


    ####### sigma = 128, L = 128, Fo = 0.0078125, lbm_iter = 49151 #######


    config.solver.corrupt_sched = np.array([1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 161, 164, 167, 170, 173, 176, 179, 183, 187, 191, 195, 199, 203, 208, 213, 218, 223, 229, 235, 241, 248, 255, 262, 270, 278, 287, 296, 306, 316, 327, 338, 350, 363, 376, 390, 405, 421, 438, 456, 474, 494, 515, 537, 560, 584, 610, 637, 666, 696, 728, 762, 798, 836])

    niu_sched = [9.97138817e-05, 1.05429356e-04, 1.11472434e-04, 1.17861894e-04,
                1.24617590e-04, 1.31760514e-04, 1.39312861e-04, 1.47298100e-04,
                1.55741042e-04, 1.64667923e-04, 1.74106481e-04, 1.84086046e-04,
                1.94637627e-04, 2.05794012e-04, 2.17589866e-04, 2.30061844e-04,
                2.43248700e-04, 2.57191410e-04, 2.71933299e-04, 2.87520174e-04,
                3.04000469e-04, 3.21425395e-04, 3.39849095e-04, 3.59328818e-04,
                3.79925095e-04, 4.01701924e-04, 4.24726974e-04, 4.49071791e-04,
                4.74812022e-04, 5.02027651e-04, 5.30803246e-04, 5.61228221e-04,
                5.93397118e-04, 6.27409896e-04, 6.63372244e-04, 7.01395908e-04,
                7.41599041e-04, 7.84106567e-04, 8.29050570e-04, 8.76570708e-04,
                9.26814641e-04, 9.79938492e-04, 1.03610734e-03, 1.09549571e-03,
                1.15828815e-03, 1.22467977e-03, 1.29487688e-03, 1.36909759e-03,
                1.44757255e-03, 1.53054559e-03, 1.61827454e-03, 1.71103200e-03,
                1.80910621e-03, 1.91280191e-03, 2.02244131e-03, 2.13836511e-03,
                2.26093351e-03, 2.39052738e-03, 2.52754941e-03, 2.67242537e-03,
                2.82560543e-03, 2.98756559e-03, 3.15880910e-03, 3.33986807e-03,
                3.53130511e-03, 3.73371509e-03, 3.94772695e-03, 4.17400570e-03,
                4.41325447e-03, 4.66621667e-03, 4.93367835e-03, 5.21647060e-03,
                5.51547215e-03, 5.83161209e-03, 6.16587277e-03, 6.51929285e-03,
                6.89297053e-03, 7.28806695e-03, 7.70580980e-03, 8.14749714e-03,
                8.61450145e-03, 9.10827385e-03, 9.63034868e-03, 1.01823482e-02,
                1.07659876e-02, 1.13830805e-02, 1.20355444e-02, 1.27254066e-02,
                1.34548110e-02, 1.42260238e-02, 1.50414417e-02, 1.59035982e-02,
                1.68151725e-02, 1.77789971e-02, 1.87980669e-02, 1.98755485e-02,
                2.10147899e-02, 2.22193313e-02, 2.34929155e-02, 2.48394999e-02,
                2.62632689e-02, 2.77686465e-02, 2.93603104e-02, 3.10432065e-02,
                3.28225642e-02, 3.47039123e-02, 3.66930970e-02, 3.87962993e-02,
                4.10200544e-02, 4.33712724e-02, 4.58572593e-02, 4.84857398e-02,
                5.12648816e-02, 5.42033202e-02, 5.73101865e-02, 6.05951344e-02,
                6.40683714e-02, 6.77406900e-02, 7.16235013e-02, 7.57288705e-02,
                8.00695543e-02, 8.46590406e-02, 8.95115905e-02, 9.46422825e-02,
                1.00067059e-01, 1.05802778e-01, 1.11867260e-01, 1.18279351e-01,
                1.25058975e-01, 1.32227198e-01, 1.39806295e-01, 1.47819816e-01,
                1.56292663e-01, 1.65251162e-01, 8.73615754e-02, 9.23690313e-02,
                9.76635082e-02, 1.03261458e-01, 1.09180275e-01, 1.15438351e-01,
                1.22055133e-01, 1.29051180e-01, 1.36448231e-01, 1.44269272e-01,
                1.52538606e-01, 1.61281927e-01, 1.13684270e-01, 1.20200510e-01,
                1.27090252e-01, 1.34374905e-01, 1.42077106e-01, 1.50220788e-01,
                1.58831255e-01, 1.25951447e-01, 1.33170826e-01, 1.40804010e-01,
                1.48874719e-01, 1.57408031e-01, 1.66430462e-01, 1.40776038e-01,
                1.48845144e-01, 1.57376760e-01, 1.66397399e-01, 1.46612574e-01,
                1.55016222e-01, 1.63901558e-01, 1.48539591e-01, 1.57053694e-01,
                1.66055814e-01, 1.53627184e-01, 1.62432902e-01, 1.52660757e-01,
                1.61411080e-01, 1.53596664e-01, 1.62400632e-01, 1.56099302e-01,
                1.65046718e-01, 1.59964740e-01, 1.56123432e-01, 1.65072231e-01,
                1.62067252e-01, 1.59932961e-01, 1.58531360e-01, 1.57758286e-01,
                1.57534082e-01, 1.66563738e-01, 1.58499866e-01, 1.59604647e-01,
                1.61082395e-01, 1.62910416e-01, 1.65071227e-01, 1.61107294e-01,
                1.64032808e-01, 1.61473931e-01, 1.65038433e-01, 1.63592090e-01,
                1.62794337e-01, 1.62562976e-01, 1.62834519e-01]


    #######################################################################

    # config.solver.corrupt_sched = conf_utils.exp_schedule(1, config.solver.final_lbm_step, config.solver.max_fwd_steps, dtype=int)
    # niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    # niu_sched = conf_utils.lin_schedule(1 / 6,  1 / 6, solver.final_lbm_step, dtype=np.float32)
    # niu_sched = conf_utils.tanh_schedule(1E-4 *1./ 6,  1./ 6, solver.max_fwd_steps, dtype=np.float32)
    
    #######################################################################


    config.solver.niu = config.solver.bulk_visc = np.array(niu_sched)
    config.solver.hash = conf_utils.hash_solver(config.solver)


    are_steps_unique = False
    if are_steps_unique:
        config.solver.corrupt_sched = np.unique(config.solver.corrupt_sched)
        config.solver.max_fwd_steps = len(config.solver.corrupt_sched)
        config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1

    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    """)
            f.write(
    """

    return config
    """)


if __name__ == '__main__':
    main()
