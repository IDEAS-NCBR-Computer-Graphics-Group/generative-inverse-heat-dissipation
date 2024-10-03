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
from pathlib import Path

import cv2 # conda install conda-forge::opencv || pip install opencv-python

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range, standarize_grayscale_image_range
from numerical_solvers.visualization.taichi_lbm_gui import run_with_gui

from configs.mnist.small_mnist_lbm_ns_config import get_config


from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter

# from lbm_diffuser.lbm_bckp_with_fields import lbm_solver as lbm_solver_bkcp
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver

# %% read IC
# https://github.com/taichi-dev/image-processing-with-taichi/blob/main/image_transpose.py

# img_path = './numerical_solvers/runners/mnist-2.png'
img_path = './numerical_solvers/runners/cat_256x256.jpg'

target_size=None
# target_size=(512, 512)
target_size = (256, 256) # None
# target_size = (64, 64) # None
# target_size = (28, 28) # None





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

    grid_size = np_gray_image.shape

    config = get_config()

    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            config.solver.domain_size, grid_size, 
            config.solver.turb_intensity, config.solver.noise_limiter,
            energy_spectrum=config.solver.energy_spectrum, 
            frequency_range={'k_min': config.solver.k_min, 'k_max': config.solver.k_max}, 
            dt_turb=config.solver.dt_turb, 
            is_div_free=False)

    solver = LBM_NS_Solver(
        "miau",
        grid_size,
        config.solver.niu, config.solver.bulk_visc,
        spectralTurbulenceGenerator
    )


    # spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
    #     domain_size, grid_size, 
    #     turb_intensity, noise_limiter,
    #     energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
    #     dt_turb=dt_turb, 
    #     is_div_free = False)
    
    
    # niu = 1E0 * 1./6
    # bulk_visc = 1E0 * 1./6
    # case_name="miau"   
    # solver = LBM_NS_Solver(
    #     case_name,
    #     np_gray_image.shape,
    #     niu, bulk_visc,
    #     spectralTurbulenceGenerator
    #     )
    
    solver.init(np_gray_image) 



    ######################################################################################################### TODO Code with Michal's renderer


    # window = ti.ui.Window('CG - Renderer', res=(5*solver.nx, 3 * solver.ny))
    # gui = window.get_gui()
    # canvas = window.get_canvas()
    
    # canvasPlotter = CanvasPlotter(solver, (1.0*np_gray_image.min(), 1.0*np_gray_image.max()))

    # # warm up
    # solver.solve(iterations=1)
    # solver.iterations_counter=0 # reset counter
    # img = canvasPlotter.make_frame()
    
    # # os.Path("output/").mkdir(parents=True, exist_ok=True)
    # # canvasPlotter.write_canvas_to_file(img, f'output/iteration_{solver.iterations_counter}.jpg')
       
    # iter_per_frame = 1
    # i = 0
    # while window.running:
    #     with gui.sub_window('MAIN MENU', x=0, y=0, width=1.0, height=0.3):
    #         iter_per_frame = gui.slider_int('steps', iter_per_frame, 1, 20)
    #         gui.text(f'iteration: {solver.iterations_counter}')
    #         if gui.button('solve'):
    #             solver.solve(iter_per_frame)      
    #             img = canvasPlotter.make_frame()
    #             # save_png(save_dir, torch_image, "s.png")
    #             i += iter_per_frame

    
    #     canvas.set_image(img.astype(np.float32))
    #     window.show()



    ##########################################################################################################

    # solver.init(1.*np.ones(grid_size, dtype=np.float32))
    # solver.creatmin_init_gray_scalerbar()
    #     ax = plt.gca()
    #     ax.set_xlim([0, nx])
    #     ax.set_ylim([0, ny])
    #     plt.grid()
    #     plt.title(f'After {(i+1)*subiterations} iterations')
    #     plt.show()
    #     cv2.imwrite(f'output/{case_name}_at_{i*subiterations}.jpg', rho_cpu)

    
    #########################33 TODO back standard renderer with multiple subwindows

    
    run_with_gui(solver, np_gray_image, iter_per_frame = 1)



    ############################


# %%
