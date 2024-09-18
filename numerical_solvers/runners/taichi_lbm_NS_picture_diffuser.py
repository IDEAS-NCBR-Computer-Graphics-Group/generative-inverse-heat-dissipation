# %% imports
import sys, os
import numpy as np

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

import taichi as ti
import taichi.math as tm
import itertools

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from numerical_solvers.visualization.taichi_lbm_gui import run_with_gui

from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver

def main():

    img_path = './numerical_solvers/runners/cat_768x768.jpg'

    target_size=None
    # target_size=(512, 512)
    target_size = (256, 256) # None
    # target_size = (128, 128) # None

    np_gray_image = read_img_in_grayscale(img_path, target_size)
    np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.95, 1.05)

    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    turb_intensity = 1E-4
    noise_limiter = (-1E-3, 1E-3)
    dt_turb = 3E-3

    # turb_intensity = 1E-3
    # energy_spectrum = lambda k: np.where(np.isinf(k), 0, k)

    energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    frequency_range = {'k_min': 2.0 * np.pi / min(domain_size), 
                       'k_max': 2.0 * np.pi / (min(domain_size) / 1024)}
    
    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        domain_size,
        grid_size, 
        turb_intensity,
        noise_limiter,
        energy_spectrum=energy_spectrum,
        frequency_range=frequency_range, 
        dt_turb=dt_turb, 
        is_div_free = False
        )

    niu = 0.5*1/6
    bulk_visc = 0.5*1/6
    case_name="mnist"   
    solver = LBM_NS_Solver(
        case_name,
        np_gray_image.shape,
        niu,
        bulk_visc,
        spectralTurbulenceGenerator
        )
    
    solver.init(np_gray_image) 

    run_with_gui(solver, np_gray_image, iter_per_frame=1)

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    main()
