#%% import stufff
import numpy as np
import torch
import torch
import torch.nn as nn
import taichi as ti
from model_code.utils import DCTBlur
from numerical_solvers.corruptors.LBM_ADE_Corruptor import LBM_ADE_Corruptor
from tests.configs.ffhq_128_lbm_ade_example import get_config
from configs import conf_utils, match_sim_numbers
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def load_image(config, L):
    img_path = './numerical_solvers/runners/cat_768x768.jpg'
    np_gray_image = read_img_in_grayscale(img_path, (L, L))
    np_gray_image = normalize_grayscale_image_range(np_gray_image, 
                                                    config.solver.min_init_gray_scale, 
                                                    config.solver.max_init_gray_scale)
    return np_gray_image

def main():
    config = get_config()
    model = config.model

    model.K = 16
    model.blur_sigma_max = 16
    model.blur_sigma_min = 1e-5
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min), np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))

    L = config.data.image_size
    np_gray_image = load_image(config, L)

    match_sim_numbers.plot_matrix(np_gray_image, title="IC")
    t_init = torch.from_numpy(np_gray_image).unsqueeze(0)

    # DCT
    dctBlur = DCTBlur(model.blur_schedule, image_size=config.data.image_size, device="cpu")
    print(model.blur_schedule)
    
    blurred_by_dct = []
    for i in range(model.K):
        fwd_steps = i * torch.ones(1, dtype=torch.long)
        dct_blurred = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
        min_val = dct_blurred.min()
        max_val = dct_blurred.max()
        blurred_by_dct.append((dct_blurred - min_val) / (max_val - min_val))
    match_sim_numbers.plot_matrices_in_grid(blurred_by_dct, columns=4)

    # LBM
    config.solver.n_denoising_steps = model.K
    config.solver.max_fwd_steps = config.solver.n_denoising_steps + 1 
    
    final_lbm_step_global = 10000
    niu_min_global = 1e-8 * 1/6
    niu_max_global = 1./6
    # we can scale down the space by oing this with subsets of the schedule and matching the values there with appropriate schaling of the searched space. 

    best = {
        'niu_min': None,
        'niu_max': None,
        'final_lbm_step': None,
        'error': np.inf
        }

    err = np.inf
    thres = 10e-4
    while err > thres:
    
        # pick values from specified range
        niu_min = random.uniform(niu_min_global, niu_max_global)
        niu_max = random.uniform(niu_min, niu_max_global)
        config.solver.final_lbm_step = random.randint(config.solver.min_fwd_steps, final_lbm_step_global)

        # update solver config
        config.solver.corrupt_sched = conf_utils.exp_schedule(
            config.solver.min_fwd_steps,
            config.solver.final_lbm_step,
            config.solver.max_fwd_steps,
            dtype=int
            )
        niu_sched = conf_utils.tanh_schedule(
            niu_min,
            niu_max,
            config.solver.final_lbm_step,
            dtype=np.float32
            )
        config.solver.niu = config.solver.bulk_visc = niu_sched
        config.solver.cs2 = conf_utils.lin_schedule(
            1./3,
            1./3,
            config.solver.final_lbm_step,
            dtype=np.float32
            )
        config.turbulence.turb_intensity = conf_utils.lin_schedule(
            0,
            0,
            config.solver.final_lbm_step,
            dtype=np.float32
            )
        
        corruptor = LBM_ADE_Corruptor(config=config, transform=config.data.transform)
        corruptor._corrupt(t_init, config.solver.n_denoising_steps)
        lbm_corrupted_images = [i.squeeze(0).detach().float().numpy() for i in corruptor.intermediate_samples[1:]]

        if err < best['error']:
            best['niu_min'] = niu_min
            best['niu_max'] = niu_max
            best['final_lbm_step'] = config.solver.final_lbm_step
            best['error'] = err
            print('Switching to new found best')
            print(best)
            match_sim_numbers.plot_matrices_in_grid(lbm_corrupted_images, columns=4)

        # calculate mse between the two lists
        total_mse = 0
        for img1, img2 in zip(blurred_by_dct, lbm_corrupted_images):
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
        
            mse = np.mean((img1 - img2) ** 2)
            total_mse += mse
        
        err = total_mse / len(blurred_by_dct)
        print(f'Best error: {best["error"]:2.6e}, error now: {err:2.6e}')
    match_sim_numbers.plot_matrices_in_grid(blurred_by_dct, columns=4)

if __name__ == '__main__':
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    main()
