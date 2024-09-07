# %% imports

# Fluid solver based on lattice boltzmann method using taichi language
# Inspired by: https://github.com/hietwll/LBM_Taichi

import sys, os
import numpy as np

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

import taichi as ti
import taichi.math as tm
import itertools

import cv2 # conda install conda-forge::opencv || pip install opencv-python

from lbm_diffuser.synthetic_turbulence import SpectralTurbulenceGenerator
from lbm_diffuser.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from lbm_diffuser.lbm_gui_visualizer import run_with_gui


# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py


img_path = 'cat_768x768.jpg'
# img_path = 'cat_768x768.jpg'
target_size=None
# target_size=(512, 512)
target_size = (256, 256) # None
# target_size = (128, 128) # None

np_gray_image = read_img_in_grayscale(img_path, target_size)
np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.95, 1.05)

# print(np_gray_image.shape)
# plt.imshow(np_gray_image, cmap='gist_gray')
# plt.colorbar()
# plt.title(f'image')
# plt.show()

# %% run sovler

# from lbm_diffuser.lbm_bckp_with_fields import lbm_solver as lbm_solver_bkcp
# from lbm_diffuser.lbm_solver_old import lbm_solver
from lbm_diffuser.LBM_ADE_Solver import LBM_ADE_Solver
             
ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)
ti_float_precision = ti.f32
  
if __name__ == '__main__':    
    nx, ny = np_gray_image.shape # 768, 768 
    niu = 1E-3*1/6
    bulk_visc = None
    
    case_name="miau"
    
    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    noise_limiter = (-1E-1, 1E-1)
    dt_turb = 1E-4

    # turb_intensity = 1E-3
    # energy_spectrum = lambda k: np.where(np.isinf(k), 0, k)
    
    turb_intensity = 8E-4
    energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    frequency_range = {'k_min': 2.0 * np.pi / min(domain_size), 
                       'k_max': 2.0 * np.pi / (min(domain_size) / 2048)}
    
    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        domain_size, grid_size, 
        turb_intensity, noise_limiter,
        energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
        dt_turb=dt_turb, 
        is_div_free=False)
        
    solver = LBM_ADE_Solver(
        case_name,
        np_gray_image.shape,
        niu, bulk_visc,
        spectralTurbulenceGenerator
        )
    
    # 
    solver.init(np_gray_image) 

    # solver.init(1.*np.ones((nx,ny), dtype=float))
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

    
    
    run_with_gui(solver, np_gray_image, iter_per_frame=5)


# %%
