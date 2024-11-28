#%% import stufff
import numpy as np
from numba import jit, njit
from numpy.testing import assert_almost_equal
from numpy.fft import fft2, fftshift, ifft2
import matplotlib
import matplotlib.pyplot as plt
import time
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms import GaussianBlur
import torch.nn as nn
import math
import taichi as ti
import taichi.math as tm
from model_code.utils import DCTBlur
from numerical_solvers.solvers.LBM_ADE_Solver import LBM_ADE_Solver
from numerical_solvers.corruptors.LBM_ADE_Corruptor import LBM_ADE_Corruptor

from tests.configs.ffhq_128_lbm_ade_example import get_config
# from configs.mnist.small_mnist_lbm_ade_turb_config import get_config
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter
from configs import conf_utils, match_sim_numbers
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range

def load_image(config, L):
    img_path = './numerical_solvers/runners/cat_768x768.jpg'
    np_gray_image = read_img_in_grayscale(img_path, (L, L))
    np_gray_image = normalize_grayscale_image_range(np_gray_image, 
                                                    config.solver.min_init_gray_scale, 
                                                    config.solver.max_init_gray_scale)
    return np_gray_image

def display_schedules(Fo, Fo_realizable):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    # plt.plot(niu_array,  'rx', label=f'niu_array')
    plt.plot(Fo,  'rx', label=f'Fo_schedule')
    plt.plot(Fo_realizable,  'gx', label=f'Fo_realizable')
    # plt.plot(corrupt_sched, Fo_realizable, 'bx', label=f'Fo_realizable(lbm_steps)')
    # plt.plot(np.unique(Fo_schedule_unique), 'bx', label=f'Fo_schedule_unique')


    ax.grid(True, which="both", ls="--")
    ax.set_xlabel(r"denoising steps")
    ax.set_ylabel(r"Fo")

    plt.legend()


def main():
    config = get_config()
    model = config.model

    model.K = 20
    model.blur_sigma_max = 32
    model.blur_sigma_min = 0.001
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min), np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))

    dctBlur = DCTBlur(model.blur_schedule, image_size=config.data.image_size, device="cpu")
    L = config.data.image_size
    np_gray_image = load_image(config, L)
    np_gray_image_rot90 = np.rot90(np_gray_image.copy(), k=-1)

    match_sim_numbers.plot_matrix(np_gray_image, title="IC")
    t_initial_condition = torch.from_numpy(np_gray_image).unsqueeze(0)
    t_init = torch.from_numpy(np_gray_image).unsqueeze(0)

    fwd_steps = model.K * torch.ones(1, dtype=torch.long)
    blurred_by_dct = dctBlur(t_initial_condition, fwd_steps).float().squeeze().numpy()
    match_sim_numbers.plot_matrix(blurred_by_dct, title="DCT schedule Blurr")

    Fo_array = match_sim_numbers.calc_Fo(model.blur_schedule, L)

    niu_min = 1E-4 * 1/6
    niu_max = 1./6

    (dt_array, 
     niu_array, 
     niu_array_realizable, 
     dFo_realizable) = match_sim_numbers.calculate_t_niu_array(Fo_array, niu_min, niu_max, L)

    Fo_realizable =  np.array(np.cumsum(dFo_realizable))
    corrupt_sched =  np.array(list(np.cumsum(dt_array)), dtype=int) # scan add
    print(f"niu_array = {niu_array} \n\n niu_array_realizable = {niu_array_realizable}")
    print(f"sigma = {model.blur_schedule} \n\n Fo={Fo_array} \n\n dt_array = {dt_array}, \n\n corrupt_sched = {corrupt_sched}")

    config.solver.max_fwd_steps = model.K
    config.solver.n_denoising_steps = model.K - 1
    config.solver.corrupt_sched = corrupt_sched

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32)
    config.turbulence.turb_intensity = conf_utils.lin_schedule(0, 0, corrupt_sched[-1], dtype=np.float32)
    config.solver.niu = config.solver_bulk_visc = np.array(sum([[niu_array[i]]*dt_array[i] for i in range(len(niu_array))], []))
    config.solver.final_lbm_step = int(corrupt_sched[-1])

    spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
        config.turbulence.domain_size,
        (L, L), 
        0 * config.turbulence.turb_intensity,
        config.turbulence.noise_limiter,
        energy_spectrum=config.turbulence.energy_spectrum, 
        frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
        dt_turb=config.turbulence.dt_turb, 
        is_div_free=False
        )

    for i in range(config.solver.max_fwd_steps):
        niu_array = np.array([niu_array[i]]*corrupt_sched[i])

        solver = LBM_ADE_Solver(
            (L, L),
            niu_array,
            niu_array,
            conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32),
            spectralTurbulenceGenerator
            )
        solver.init(np_gray_image_rot90) 

        for lbm_iter in dt_array:
            solver.solve(iterations=lbm_iter)
            
        # solver.solve(iterations=sum(dt_array))
        img = solver.rho
        blurred_by_lbm = np.rot90(img.to_numpy().copy(), k=1) # torch to numpy + rotation

    match_sim_numbers.plot_matrix_side_by_side(blurred_by_lbm, blurred_by_dct, title="LBM BLURR")
    match_sim_numbers.plot_matrix(blurred_by_dct-blurred_by_lbm, title="Difference DCT schedule - LBM")
    print(f'Difference: {abs(blurred_by_dct - blurred_by_lbm)}')
    # print(f'Maximal value of DCT blurr: {blurred_by_dct.max()}')
    # print(f'Maximal value of LBM blurr: {blurred_by_lbm.max()}')

if __name__ == '__main__':
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    main()