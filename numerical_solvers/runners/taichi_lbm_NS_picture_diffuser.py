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


from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter

# from lbm_diffuser.lbm_bckp_with_fields import lbm_solver as lbm_solver_bkcp
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver

# from numerical_solvers.solvers.LBM_NS_Solver_OLD import LBM_NS_Solver_OLD as LBM_NS_Solver

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py

# img_path = './numerical_solvers/runners/mnist-2.png'
# img_path = './numerical_solvers/runners/cat_256x256.jpg'
img_path = './numerical_solvers/runners/cat_768x768.jpg'
target_size=None
# target_size=(512, 512)
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

             
ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu)
ti_float_precision = ti.f64
  
if __name__ == '__main__':    

    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    turb_intensity = 0* 1E-4
    noise_limiter = (-1E-3, 1E-3)
    dt_turb = 1E-3 


    # energy_spectrum = lambda k: np.where(np.isinf(k), 0, k)
    
    energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    frequency_range = {'k_min': 2.0 * torch.pi / min(domain_size),
                       'k_max': 2.0 * torch.pi / (min(domain_size) / 1024)}
    
    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        domain_size, grid_size, 
        turb_intensity, noise_limiter,
        energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
        dt_turb=dt_turb, 
        is_div_free = False)
    
    
    # niu = 0.00001 * 1./6
    niu = 1./6
    bulk_visc = niu
    
    solver = LBM_NS_Solver(
        grid_size,
        niu, bulk_visc,
        spectralTurbulenceGenerator
        )
    
    solver.init(np_gray_image)


    ######################################################################################################### TODO Code with Michal's renderer


    solver.init(1.*np.ones(grid_size, dtype=np.float32))
    
    solver.create_ic_hill(.9, 1E-2, int(0.5*grid_size[0]), int(0.*grid_size[1]))
    
    
    # solver.create_ic_hill(.5, 1E-2, int(0.5*grid_size[0]), int(0.5*grid_size[1])) 
    # solver.create_ic_hill(.05, 1E-3, int(0.25*grid_size[0]), int(0.25*grid_size[1]))
    # solver.create_ic_hill(-.05, 1E-3,int(0.75*grid_size[0]), int(0.75*grid_size[1]))
    
    
    output_dir = "output_dp09_nu1by6"
    os.makedirs(output_dir, exist_ok=True)
    matplotlib.use('TkAgg')
    for i in range(48):
        subiterations = 25
        solver.solve(subiterations)
        rho_cpu = solver.rho.to_numpy()

        plt.imshow(rho_cpu, vmin=0.95, vmax=1.05, cmap="gist_gray", interpolation='none') 
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlim([0, grid_size[0]])
        ax.set_ylim([0, grid_size[1]])
        plt.grid()
        plt.title(f'After {(i+1)*subiterations} iterations')
        plt.savefig(f'{output_dir}/rho_at_{i*subiterations}.jpg')  # Save with Matplotlib
        plt.close()
        # plt.show()
        
        # cv2.imwrite(f'output/rho_at_{i*subiterations}.jpg', rho_cpu)

    
    #########################33 TODO back standard renderer with multiple subwindows


    # run_with_gui(solver, np_gray_image, iter_per_frame = 1)



    ############################


# %%
