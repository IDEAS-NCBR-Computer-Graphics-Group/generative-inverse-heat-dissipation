# %% imports

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

import sys, os
import numpy as np

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import torch

import taichi as ti
import taichi.math as tm
import itertools

import cv2 # conda install conda-forge::opencv || pip install opencv-python

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from numerical_solvers.visualization.taichi_lbm_gui import run_with_gui

from configs.mnist.small_mnist_lbm_ade_turb_config import get_config

from numerical_solvers.solvers.LBM_ADE_Solver import LBM_ADE_Solver

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py


img_path = './numerical_solvers/runners/cat_768x768.jpg'
# target_size=None
# target_size=(512, 512)
target_size = (256, 256) 
# target_size = (128, 128) 

np_gray_image = read_img_in_grayscale(img_path, target_size)
np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.95, 1.05)

# print(np_gray_image.shape)
# plt.imshow(np_gray_image, cmap='gist_gray')
# plt.colorbar()
# plt.title(f'image')
# plt.show()

# %% run sovler

             
is_gpu_avail = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
print(f"device = {device}")
    
ti_float_precision = ti.f32
  
if __name__ == '__main__':    

    case_name="miau"
    config = get_config()

    grid_size = np_gray_image.shape


    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(config.turbulence.domain_size, grid_size, 
            config.turbulence.turb_intensity, config.turbulence.noise_limiter,
            energy_spectrum=config.turbulence.energy_spectrum, 
            frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
            dt_turb=config.turbulence.dt_turb, 
        is_div_free=False)
    

    solver = LBM_ADE_Solver(
        np_gray_image.shape,
        config.solver.niu, config.solver.bulk_visc, config.solver.cs2,
        spectralTurbulenceGenerator
        )    
    # 
    solver.init(np_gray_image) 

    run_with_gui(solver, np_gray_image, iter_per_frame=100)
