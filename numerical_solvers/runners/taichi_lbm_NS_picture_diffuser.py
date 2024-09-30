from absl import app, flags
import numpy as np
from ml_collections.config_flags import config_flags
import torch
import taichi as ti
import matplotlib.pyplot as plt

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

img_path = './numerical_solvers/runners/mnist-2.png'
# img_path = './numerical_solvers/runners/cat_256x256.jpg'

target_size=None
# target_size=(512, 512)
target_size = (256, 256) # None
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

    domain_size = (1.0, 1.0)
    grid_size = np_gray_image.shape
    turb_intensity = 0*1E-4

    noise_limiter = (-1E-3, 1E-3)
    dt_turb = 1E-3 

    # turb_intensity = 1E-4
    # energy_spectrum = lambda k: np.where(np.isinf(k), 0, k)
    
    energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    frequency_range = {'k_min': 2.0 * np.pi / min(domain_size), 
                       'k_max': 2.0 * np.pi / (min(domain_size) / 1024)}
    
from PIL import Image
import torchvision
import os

from solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range, make_grayscale
from solvers.LBM_NS_Solver import LBM_NS_Solver
from corruptors.CorruptedDatasetCreator import preprocess_dataset
from scripts.utils import save_png_norm, save_png
from scripts.datasets import get_dataset
from visualization.CanvasPlotter import CanvasPlotter

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])


def run_corruption(config):
    target_size = (256, 256) # None

    # np_gray_image = read_img_in_grayscale(img_path, target_size)
    trainloader, testloader = get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    torch_image = next(iter(trainloader))[0][0]
    storage_dir = 'runs'
    save_scriptname = 'corruption_experiments'
    save_dir = 'runs/corrupt_image'
    os.makedirs(save_dir, exist_ok=True)
    pil_image = torchvision.transforms.ToPILImage()(torch_image)
    np_gray_image = np.array(pil_image)
    np_gray_image = make_grayscale(np_gray_image)
    np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.95, 1.05)
    np_gray_image = np.rot90(np_gray_image, -1)

    case_name="mnist"   

    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        domain_size, grid_size, 
        turb_intensity, noise_limiter,
        energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
        dt_turb=dt_turb, 
        is_div_free = False)
    
    
    niu = 1 * 1./6
    bulk_visc = 1 * 1./6
    case_name="miau"   
    
        config.lbm.solver.domain_size,
        [config.data.image_size]*2, 
        config.lbm.solver.turb_intensity,
        config.lbm.solver.noise_limiter,
        energy_spectrum=config.lbm.solver.energy_spectrum,
        frequency_range={'k_min': config.lbm.solver.k_min, 'k_max': config.lbm.solver.k_max}, 
        dt_turb=config.lbm.solver.dt_turb, 
        is_div_free = False,
        device='cuda'
        )

    solver = LBM_NS_Solver(
        case_name,
        [config.data.image_size]*2,
        config.lbm.solver.niu,
        config.lbm.solver.bulk_visc,
        spectralTurbulenceGenerator
        )
    
    # solver.init(np_gray_image) 

    # solver.init(1.*np.ones(grid_size, dtype=np.float32))
    # solver.create_ic_hill(.2, 1E-2, int(0.5*grid_size[0]), int(0.5*grid_size[1])) 
    # solver.create_ic_hill(.05, 1E-3, int(0.25*grid_size[0]), int(0.25*grid_size[1]))
    # solver.create_ic_hill(-.05, 1E-3,int(0.75*grid_size[0]), int(0.75*grid_size[1]))

    solver.init(1.*np.ones(grid_size, dtype=np.float32))
    solver.create_ic_hill(0.5, 1E-3, int(0.5*grid_size[0]), int(0.5*grid_size[1])) 
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

    
    #########################33 TODO back standard renderer with multiple subwindows

    
    run_with_gui(solver, np_gray_image, iter_per_frame = 5)



    ############################


# %%
    iter_per_frame = 1
    window = ti.ui.Window('CG - Renderer', res=(2*solver.nx, 3 * solver.ny))
    gui = window.get_gui()
    canvas = window.get_canvas()
    
    canvasPlotter = CanvasPlotter(solver, (1.0*np_gray_image.min(), 1.0*np_gray_image.max()))

    # warm up
    solver.solve(iterations=1)   
    solver.iterations_counter=0 # reset counter
    img = canvasPlotter.make_frame()
    
    # os.Path("output/").mkdir(parents=True, exist_ok=True)
    # canvasPlotter.write_canvas_to_file(img, f'output/iteration_{solver.iterations_counter}.jpg')
       
    i = 0
    while window.running:
        with gui.sub_window('MAIN MENU', x=0, y=0, width=1.0, height=0.3):
            iter_per_frame = gui.slider_int('steps', iter_per_frame, 1, 20)
            gui.text(f'iteration: {solver.iterations_counter}')
            if gui.button('solve'):
                solver.solve(iter_per_frame)      
                img = canvasPlotter.make_frame()
                save_png(save_dir, torch_image, "s.png")
                i += iter_per_frame

    
        canvas.set_image(img.astype(np.float32))
        window.show()


def main(argv):
    ti.init(arch=ti.gpu)
    run_corruption(FLAGS.config)

if __name__ == '__main__':
    app.run(main)
