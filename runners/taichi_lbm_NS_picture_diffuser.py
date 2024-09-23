from absl import app, flags
import numpy as np
from ml_collections.config_flags import config_flags
import torch
import taichi as ti
import matplotlib.pyplot as plt
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
    # (fluid_train, fluid_test), (blur_train, blur_test) = preprocess_dataset(config)

    # img_path = './numerical_solvers/runners/cat_768x768.jpg'

    # target_size=None
    # target_size=(512, 512)
    target_size = (256, 256) # None
    # target_size = (128, 128) # None

    # np_gray_image = read_img_in_grayscale(img_path, target_size)
    trainloader, testloader = get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    torch_image = next(iter(trainloader))[0][0]

    save_dir = 'runs/corrupt_image'
    os.makedirs(save_dir, exist_ok=True)

    pil_image = torchvision.transforms.ToPILImage()(torch_image)
    
    np_gray_image = np.array(pil_image)
    # print(np_gray_image.shape)
    # xy = np_gray_image.shape[-2]
    # np_gray_image = torch_gray_image.numpy().reshape((xy,xy))
    # print(np_gray_image.shape)
    # np_gray_image = np.array(Image.fromarray(np_gray_image).convert('L'))
    # print(np_gray_image.shape)

    np_gray_image = make_grayscale(np_gray_image)

    np_gray_image = normalize_grayscale_image_range(np_gray_image, 0.95, 1.05)
    np_gray_image = np.rot90(np_gray_image, -1)

    domain_size = (1.0, 1.0)
    print(domain_size)
    grid_size = np_gray_image.shape
    print(grid_size)
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
        grid_size,
        niu,
        bulk_visc,
        spectralTurbulenceGenerator
        )
    
    solver.init(np_gray_image) 

    iter_per_frame=1
    window = ti.ui.Window('CG - Renderer', res=(2*solver.nx, 3 * solver.ny))
    gui = window.get_gui()
    canvas = window.get_canvas()
    
    canvasPlotter = CanvasPlotter(solver, (1.0*np_gray_image.min(), 1.0*np_gray_image.max()))
    

    # warm up
    solver.solve(iterations=1)   
    solver.iterations_counter=0 # reset counter
    img = canvasPlotter.make_frame()
    
    # Path("output/").mkdir(parents=True, exist_ok=True)
    # canvasPlotter.write_canvas_to_file(img, f'output/iteration_{solver.iterations_counter}.jpg')
       
    while window.running:
        with gui.sub_window('MAIN MENU', x=0, y=0, width=1.0, height=0.3):
            iter_per_frame = gui.slider_int('steps', iter_per_frame, 1, 20)
            if gui.button('solve'):
                solver.solve(iter_per_frame)      
                img = canvasPlotter.make_frame()
                save_png(save_dir, torch_image, "s.png")

    
        canvas.set_image(img.astype(np.float32))
        window.show()


def main(argv):
    ti.init(arch=ti.gpu)
    run_corruption(FLAGS.config)

if __name__ == '__main__':
    app.run(main)
