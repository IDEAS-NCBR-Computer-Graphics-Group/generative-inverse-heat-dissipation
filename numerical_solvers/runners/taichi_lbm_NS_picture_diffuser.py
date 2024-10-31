# %% imports

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

import sys, os
import torch
from timeit import default_timer as timer
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import taichi as ti
import taichi.math as tm
import itertools

import cv2 # conda install conda-forge::opencv || pip install opencv-python

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import (read_img_in_grayscale,
                                                  normalize_grayscale_image_range,
                                                  standarize_grayscale_image_range)
from numerical_solvers.visualization.taichi_lbm_gui import run_with_gui, run_simple_gui

from configs.lbm_ns_gui_conf import get_config
from configs import conf_utils

from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py

# img_path = './numerical_solvers/runners/mnist-2.png'
img_path = './numerical_solvers/runners/cat_768x768.jpg'
# img_path = './numerical_solvers/runners/ffhq_1024_00062.png'



# %% run solver

             
is_gpu_avail = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ti.init(arch=ti.gpu, default_fp=ti.f32) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
print(f"device = {device}")

# device = torch.device("cpu")
# ti.init(arch=ti.cpu, default_fp=ti.f32) 
  
if __name__ == '__main__':    
    domain_size = (1.0, 1.0)
    
    config = get_config()
    
    target_size=None
    target_size=(config.data.image_size, config.data.image_size)
    # target_size=(1024, 1024)
    # target_size=(768, 768)
    # target_size=(512, 512)
    # target_size = (256, 256)
    # target_size = (128, 128)
    # target_size = (64, 64)
    # target_size = (28, 28)
    # target_size = None

    np_gray_image = read_img_in_grayscale(img_path, target_size)
    np_gray_image = normalize_grayscale_image_range(np_gray_image, config.solver.min_init_gray_scale, config.solver.max_init_gray_scale)
    
    grid_size = np_gray_image.shape
    # print(np_gray_image.shape)
    # plt.imshow(np_gray_image, cmap='gist_gray')
    # plt.colorbar()
    # plt.title(f'image')
    # plt.show()

    grid_size = np_gray_image.shape

    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(config.turbulence.domain_size, grid_size, 
            config.turbulence.turb_intensity, config.turbulence.noise_limiter,
            energy_spectrum=config.turbulence.energy_spectrum, 
            frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
            dt_turb=config.turbulence.dt_turb, 
            is_div_free=False, device=device)

    solver = LBM_NS_Solver(
        np_gray_image.shape,
        config.solver.niu,
        config.solver.bulk_visc,
        config.solver.cs2,
        spectralTurbulenceGenerator
        )    
    
    solver.init(np_gray_image)

    # solver.init(1.*np.ones(grid_size, dtype=np.float32))
    # solver.create_ic_hill(.5, 1E-2, int(0.*grid_size[0]), int(0.5*grid_size[1]))
    # solver.create_ic_hill(.5, 1E-2, int(0.5*grid_size[0]), int(0.5*grid_size[1]))
    # solver.create_ic_hill(.05, 1E-3, int(0.25*grid_size[0]), int(0.25*grid_size[1]))
    # solver.create_ic_hill(-.05, 1E-3,int(0.75*grid_size[0]), int(0.75*grid_size[1]))
    
    # output_dir = "local_outputs/test"
    # os.makedirs(output_dir, exist_ok=True)
    # matplotlib.use('TkAgg')
    # subiterations = 1
    # start = timer()
    # for i in range(100):
    #     rho_cpu = solver.rho.to_numpy()
    #     rho_cpu = rho_cpu.T
    #     plt.imshow(rho_cpu, vmin=1. - drho, vmax= 1. + drho, cmap="gist_gray", interpolation='none')
    #     plt.colorbar()
    #     ax = plt.gca()
    #     ax.set_xlim([0, grid_size[0]])
    #     ax.set_ylim([0, grid_size[1]])
    #     plt.grid()
    #     total_iter = i*subiterations
    #     plt.title(f'After {total_iter} iterations')
    #     plt.savefig(f'{output_dir}/rho_at_{total_iter}.png')  # Save with Matplotlib
    #     # plt.show()
    #     plt.close()
    #     print(f"Savefig at iteration {total_iter}")
    #     solver.solve(subiterations)
    # end = timer()
    # print(f"Corruption took {end - start:.2f} seconds")
        
    ############################ standard renderer with multiple subwindows
    # run_with_gui(solver, np_gray_image, iter_per_frame = 1)
    run_simple_gui(solver, np_gray_image, iter_per_frame=1, sleep_time=0.075, show_gui=True) # sleep_time=0.075,
    ############################
