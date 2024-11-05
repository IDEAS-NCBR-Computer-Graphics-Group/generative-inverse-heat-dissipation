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

from conf_utils import evaluate_config_file_name

# def main():
#     # Define the save directory and make sure it exists
#     save_dir = os.path.join("configs", "campaign_ffhq_ns_128")
#     os.makedirs(save_dir, exist_ok=True)

#     # Define the default configuration location
#     default_cfg_dir_list = ["configs", "ffhq", "res_128", "ade"]
#     default_cfg_file = "default_lbm_ffhq_128_config.py"

#     # Debugging path
#     src_path = os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file)
#     print(f"Source path: {src_path}")

#     # Attempt to copy the file
#     src_path = "/raid/NFS_SHARE/home/jakub.meixner/generative-inverse-heat-dissipation/configs/ffhq/res_128/ade/default_lbm_ffhq_128_config.py"
#     shutil.copy(src_path, save_dir)


#     src_path = os.path.join("configs", "ffhq", "res_128", "ade", default_cfg_file)
#     print(f"Absolute source path: {os.path.abspath(src_path)}")

def main():
    # Define the hyperparameter grid
    save_dir = os.path.join("configs","campaign_ffhq_ade_128_timesteps_list_6")
    os.makedirs(save_dir, exist_ok=False)
    param_grid = {
        'training.batch_size': [4],
        'training.n_iters': [30001],
        'optim.lr': [2e-5],
        # 'optim.lr': [1e-4, 5e-5, 2e-5, 1e-5],
        'turbulence.turb_intensity' : [1e-4, 1e-3],
        'solver.cs2' : [1./3.]
        # 'solver.cs2' : [0.3*1./3 , 1./3 ]
    }
    
    # param_grid = {
    #     'training.batch_size': [64],
    #     'training.n_iters': [1001],
    #     'optim.lr': [2e-5, 1e-5],
    #     'turbulence.turb_intensity' : [0, 1E-4],
    #     'solver.cs2' : [0.3*1./3 , 0.6*1./3 , 1./3 ]
    # }

    default_cfg_dir_list = ["ffhq", "res_128", "ade"]
    default_cfg_file = "default_lbm_ffhq_128_config.py"
    src_path = os.path.join(os.path.join(*default_cfg_dir_list), default_cfg_file)
    print(f"Source path: {src_path}")
    
    
    shutil.copy(src_path, save_dir)

    # Create the grid
    grid = ParameterGrid(param_grid)
    # Define the directory to save the config files
    # Define the default config

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
            """)
                
            f.write(f"\n") # flush 
            
            # Write only the parameters that were changed
            for param_key, param_value in params.items():
                f.write(f"    config.{param_key} = {repr(param_value)}\n")

            f.write(f"""
    stamp = config.stamp

    timesteps_list_1 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 20, 23, 27, 32, 37, 43,
                        50, 59, 68, 79, 92, 107, 125, 145, 169, 197, 229, 266, 309, 359, 418, 486, 565, 657, 763, 887, 1032, 1199]

    timesteps_list_2 = [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 11, 14, 17, 22, 28, 35, 44, 55, 69, 86, 109, 136, 171, 215, 269, 338, 423, 531,
                    666, 836, 1048, 1314, 1648, 2067, 2592, 3250, 4076, 5111, 6410, 8038, 10080, 12640, 15851, 19877, 24926, 31257, 39196, 49151]  

    timesteps_list_3 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 24, 27, 30, 33,
                    37, 42, 47, 52, 59, 66, 74, 82, 92, 103, 115, 129, 145, 162, 181, 203, 227, 254, 284, 317, 355, 397, 444, 497, 556, 622, 696,
                    778, 871, 974, 1089, 1219, 1363, 1525, 1706, 1908, 2134, 2387, 2670, 2987, 3341, 3737, 4180, 4675, 5230, 5850, 6543, 7319, 8186,
                    9157, 10242, 11457, 12815, 14334, 16033, 17934, 20059, 22437, 25097, 28072, 31400, 35122, 39286, 43943, 49151]                
    
    timesteps_list_4 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11,
                        12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 26, 28, 31, 33, 36, 38, 41, 45, 48, 52, 56, 60, 65, 70, 76, 82, 88, 95, 102, 110,
                        119, 128, 138, 148, 160, 172, 186, 200, 216, 232, 250, 270, 291, 313, 338, 364, 392, 422, 455, 490, 528, 569, 613, 661,
                        712, 767, 826, 890, 959, 1033, 1113, 1199]

    timesteps_list_5 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
           3,     3,     3,     3,     3,     4,     4,     4,     4, 5,     5,     5,     6,     6,     6,     7,     7,     8,
           8,     9,     9,    10,    10,    11,    12,    12,    13, 14,    15,    16,    17,    17,    19,    20,    21,    22,
          23,    25,    26,    28,    29,    31,    33,    35,    37, 39,    41,    43,    46,    49,    51,    54,    57,    61,
          64,    68,    72,    76,    80,    85,    90,    95,   101, 106,   113,   119,   126,   133,   141,   149,   157,   167,
         176,   186,   197,   208,   220,   233,   246,   260,   275, 291,   308,   326,   344,   364,   385,   407,   430,   455,
         481,   509,   538,   569,   601,   636,   672,   711,   752, 795,   840,   888,   939,   993,  1050,  1110,  1174,  1241,
        1313,  1388,  1467,  1552,  1641,  1735,  1834,  1939,  2050, 2168,  2292,  2424,  2563,  2709,  2865,  3029,  3203,  3386,
        3580,  3786,  4003,  4232,  4475,  4731,  5002,  5289,  5592, 5913,  6251,  6610,  6989,  7389,  7813,  8260,  8734,  9235,
        9764, 10323, 10915, 11541, 12202, 12902, 13641, 14423, 15249, 16124, 17048, 18025, 19058, 20150, 21305, 22526, 23817, 25182,
       26625, 28151, 29765, 31471, 33274, 35182, 37198, 39330, 41584, 43967, 46487, 49151]

    timesteps_list_6 = [    0,   147,    78,    55,    44,    37,    32,    29,    27,
          25,    24,    23,    23,    22,    22,    21,    21,    21,
          21,    21,    21,    21,    21,    22,    22,    22,    23,
          23,    24,    24,    25,    25,    26,    27,    27,    28,
          29,    30,    31,    31,    32,    33,    35,    36,    37,
          38,    40,    41,    42,    44,    46,    47,    49,    51,
          53,    55,    57,    59,    61,    64,    66,    69,    72,
          75,    78,    81,    85,    88,    92,    96,   100,   104,
         108,   113,   118,   123,   128,   134,   140,   146,   153,
         159,   166,   174,   182,   190,   198,   207,   217,   227,
         237,   248,   259,   271,   284,   297,   311,   325,   340,
         356,   373,   390,   409,   428,   448,   469,   492,   515,
         540,   565,   592,   621,   650,   681,   714,   749,   785,
         823,   862,   904,   948,   994,  1043,  1093,  1147,  1203,
        1262,  1324,  1388,  1457,  1528,  1604,  1683,  1766,  1853,
        1945,  2041,  2142,  2249,  2361,  2478,  2602,  2731,  2868,
        3011,  3162,  3320,  3487,  3661,  3845,  4039,  4242,  4456,
        4680,  4916,  5165,  5426,  5700,  5989,  6292,  6611,  6947,
        7299,  7670,  8061,  8471,  8903,  9356,  9834, 10336, 10864,
       11420, 12004, 12619, 13265, 13945, 14661, 15413, 16205, 17038,
       17915, 18837, 19807, 20828, 21902, 23032, 24221, 25473, 26789,
       28175, 29633, 31167, 32782, 34481, 36270, 38152, 40132, 42217,
       44411, 46721, 49151]

    

    timesteps_array = np.array(timesteps_list_6)
    config.solver.hash = conf_utils.hash_solver(config.solver)
    config.turbulence.hash = conf_utils.hash_solver(config.turbulence)
    
    config.model.hash = conf_utils.hash_solver(config.model)
    config.optim.hash = conf_utils.hash_solver(config.optim)
    config.training.hash = conf_utils.hash_int(config.training.batch_size)
    config.solver.corrupt_sched = timesteps_array
    # config.solver.corrupt_sched = conf_utils.exp_schedule(
        # 1, 196607, config.solver.max_fwd_steps, dtype=int)

    
    
    stamp.fwd_solver_hash = conf_utils.hash_joiner([config.solver.hash, config.turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([config.model.hash, config.optim.hash, config.training.hash])
    """)

            f.write(
    """
    return config
    """)

if __name__ == '__main__':
    main()
