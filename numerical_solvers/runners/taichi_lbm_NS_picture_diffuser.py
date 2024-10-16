# %% imports

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

import sys, os
import torch

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import taichi as ti
import taichi.math as tm
import itertools

import cv2 # conda install conda-forge::opencv || pip install opencv-python

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range, standarize_grayscale_image_range
from numerical_solvers.visualization.taichi_lbm_gui import run_with_gui

from configs.mnist.small_mnist_lbm_ns_turb_config import get_config

from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter
from configs.conf_utils import lin_schedule, exp_schedule
# from lbm_diffuser.lbm_bckp_with_fields import lbm_solver as lbm_solver_bkcp
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver

# from numerical_solvers.solvers.LBM_NS_Solver_OLD import LBM_NS_Solver_OLD as LBM_NS_Solver

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py

# img_path = './numerical_solvers/runners/mnist-2.png'
# img_path = './numerical_solvers/runners/cat_256x256.jpg'
img_path = './numerical_solvers/runners/cat_768x768.jpg'
target_size=None
target_size=(512, 512)
# target_size = (256, 256)
# target_size = (128, 128)
# target_size = (64, 64)
# target_size = (28, 28)
# target_size = None




drho = 1E-1
np_gray_image = read_img_in_grayscale(img_path, target_size)
np_gray_image = normalize_grayscale_image_range(np_gray_image, 1. - drho, 1. + drho)

# print(np_gray_image.shape)
# plt.imshow(np_gray_image, cmap='gist_gray')
# plt.colorbar()
# plt.title(f'image')
# plt.show()

# %% run sovler

             
# is_gpu_avail = torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ti.init(arch=ti.gpu, default_fp=ti.f32) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
# print(f"device = {device}")

device = torch.device("cpu")
ti.init(arch=ti.cpu, default_fp=ti.f32) 
  
if __name__ == '__main__':    
    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    config = get_config()

    grid_size = np_gray_image.shape

    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(config.turbulence.domain_size, grid_size, 
            config.turbulence.turb_intensity, config.turbulence.noise_limiter,
            energy_spectrum=config.turbulence.energy_spectrum, 
            frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
            dt_turb=config.turbulence.dt_turb, 
        is_div_free=False)

    
    n = 500
    niu_sched  = exp_schedule(1E-4 * 1./6., 1./6., n)
    # niu_sched  = lin_schedule(1E-4 * 1./6., 1./6., n)
    # niu_sched  = lin_schedule(1E-4 * 1./6., 1E-4 *1./6., n)
    bulk_visc_sched = niu_sched
    
    solver = LBM_NS_Solver(
        np_gray_image.shape,
        config.solver.niu, 
        config.solver.bulk_visc,
        spectralTurbulenceGenerator
        )    
    
    solver.init(np_gray_image)

    # solver.init(1.*np.ones(grid_size, dtype=np.float32))
    # solver.create_ic_hill(.5, 1E-2, int(0.5*grid_size[0]), int(0.5*grid_size[1])) 
    # solver.create_ic_hill(.05, 1E-3, int(0.25*grid_size[0]), int(0.25*grid_size[1]))
    # solver.create_ic_hill(-.05, 1E-3,int(0.75*grid_size[0]), int(0.75*grid_size[1]))
    
    output_dir = "local_outputs/kotek"
    os.makedirs(output_dir, exist_ok=True)
    matplotlib.use('TkAgg')
    subiterations = 25
    for i in range(20):
        
        rho_cpu = solver.rho.to_numpy()
        rho_cpu = rho_cpu.T
        plt.imshow(rho_cpu, vmin=1. - drho, vmax= 1. + drho, cmap="gist_gray", interpolation='none') 
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlim([0, grid_size[0]])
        ax.set_ylim([0, grid_size[1]])
        plt.grid()
        plt.title(f'After {(i+1)*subiterations} iterations')
        plt.savefig(f'{output_dir}/rho_at_{i*subiterations}.png')  # Save with Matplotlib
        # plt.show()
        plt.close()
        
        solver.solve(subiterations)
        
    ############################ standard renderer with multiple subwindows
    # run_with_gui(solver, np_gray_image, iter_per_frame = 10)
    ############################
