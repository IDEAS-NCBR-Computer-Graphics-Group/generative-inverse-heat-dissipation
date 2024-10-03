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

             
ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)
ti_float_precision = ti.f32
  
if __name__ == '__main__':    
    nx, ny = np_gray_image.shape
    niu = 1/6
    bulk_visc = None
    
    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    noise_limiter = (-1E3, 1E3)
    dt_turb = 3E-4

    # turb_intensity = 3E-3
    # energy_spectrum = lambda k: torch.where(torch.isinf(k), 0, k)
    
    # turb_intensity = 1E-3
    # energy_spectrum = lambda k: torch.where(torch.isinf(k * k), 0, k * k) # 
    
    turb_intensity =0* 1E-4
    energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-1.)), 0, k ** (-1.0)) # najs
    
    # turb_intensity = 3E-3
    # energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    frequency_range = {'k_min': 2.0 * torch.pi / min(domain_size), 
                       'k_max': 2.0 * torch.pi / (min(domain_size) / 1024)}
    
    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        domain_size, grid_size, 
        turb_intensity, noise_limiter,
        energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
        dt_turb=dt_turb, 
        is_div_free=False)
        
    solver = LBM_ADE_Solver(
        np_gray_image.shape,
        niu, bulk_visc,
        spectralTurbulenceGenerator
        )
    
    # 
    solver.init(np_gray_image) 

    # solver.init(1.*np.ones((nx,ny), dtype=np.float32))
    # solver.create_ic_hill(.1, 1E-3, int(0.5*nx), int(0.5*ny)) 
    # solver.create_ic_hill( .05, 1E-3, int(0.25*nx), int(0.25*ny))
    # solver.create_ic_hill(-.05, 1E-3, int(0.75*nx), int(0.75*ny))
    
    # for i in range(3):
    #     subiterations = 100
    #     solver.solve(subiterations)
    #     rho_cpu = solver.rho.to_numpy()

    #     os.makedirs("output", exist_ok=True)
    #     matplotlib.use('TkAgg')
    #     plt.imshow(rho_cpu, vmin=np_gray_image.min(), vmax=np_gray_image.max(), cmap="gist_gray", interpolation='none') 
    #     plt.colorbar()
    #     ax = plt.gca()
    #     ax.set_xlim([0, nx])
    #     ax.set_ylim([0, ny])
    #     plt.grid()
    #     plt.title(f'After {(i+1)*subiterations} iterations')
    #     plt.show()
    #     cv2.imwrite(f'output/{case_name}_at_{i*subiterations}.jpg', rho_cpu)
    run_with_gui(solver, np_gray_image, iter_per_frame=100)

# %%
