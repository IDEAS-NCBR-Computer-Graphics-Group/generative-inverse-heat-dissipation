#%% import stufff
import numpy as np
import matplotlib.pyplot as plt
import torch
import taichi as ti

from model_code.utils import DCTBlur
from numerical_solvers.solvers.LBM_ADE_Solver import LBM_ADE_Solver
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from tests.configs.ffhq_128_lbm_ade_example import get_config
from configs import conf_utils, match_sim_numbers

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
    plt.plot(Fo,  'rx', label=f'Fo_schedule')
    plt.plot(Fo_realizable,  'gx', label=f'Fo_realizable')

    ax.grid(True, which="both", ls="--")
    ax.set_xlabel(r"denoising steps")
    ax.set_ylabel(r"Fo")

    plt.legend()

def main():
    config = get_config()
    model = config.model

    model.K = 50
    model.blur_sigma_max = 20
    model.blur_sigma_min = 0.5
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min), np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))

    dctBlur = DCTBlur(model.blur_schedule, image_size=config.data.image_size, device="cpu")
    L = config.data.image_size
    np_gray_image = load_image(config, L)
    np_gray_image_rot90 = np.rot90(np_gray_image.copy(), k=-1)

    match_sim_numbers.plot_matrix(np_gray_image, title="IC")
    t_initial_condition = torch.from_numpy(np_gray_image).unsqueeze(0)
    t_init = torch.from_numpy(np_gray_image).unsqueeze(0)

    fwd_steps = model.K* torch.ones(1, dtype=torch.long)
    blurred_by_dct = dctBlur(t_initial_condition, fwd_steps).float().squeeze().numpy()
    match_sim_numbers.plot_matrix(blurred_by_dct, title="DCT schedule Blurr")

    blurred_by_dct_all = []
    for i in range(model.K+1):
        fwd_steps = i * torch.ones(1, dtype=torch.long)
        dct_blurred = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
        min_val = dct_blurred.min()
        max_val = dct_blurred.max()
        blurred_by_dct_all.append((dct_blurred - min_val) / (max_val - min_val))
    match_sim_numbers.plot_matrices_in_grid(blurred_by_dct_all, columns=4)

    Fo_array = match_sim_numbers.calc_Fo(model.blur_schedule, L)

    niu_min = 1e-4 * 1./6
    niu_max = 1e-1 * 1./6

    (dt_array, 
     niu_array, 
     niu_array_realizable, 
     dFo_realizable) = match_sim_numbers.calculate_t_niu_array_from_0(Fo_array, niu_min, niu_max, L)

    Fo_realizable =  np.array(np.cumsum(dFo_realizable))
    corrupt_sched =  np.array(list(np.cumsum(dt_array)), dtype=int) # scan add
    print(f'niu_array = {niu_array}')
    print(f'niu_array_realizable = {niu_array_realizable}')
    print(f'sigma = {model.blur_schedule}')
    print(f'Fo={Fo_array}')
    print(f'dt_array = {dt_array}')
    print(f'corrupt_sched = {corrupt_sched}')

    config.solver.max_fwd_steps = model.K
    config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
    config.solver.corrupt_sched = corrupt_sched

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32)
    config.turbulence.turb_intensity = conf_utils.lin_schedule(0, 0, corrupt_sched[-1], dtype=np.float32)
    # config.solver.niu = config.solver_bulk_visc = np.array(sum([[niu_array[i]]*dt_array[i] for i in range(len(niu_array))], []))
    # config.solver.final_lbm_step = int(corrupt_sched[-1])

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

    min_val = np_gray_image.min()
    max_val = np_gray_image.max()
    np_gray_image_min_max = (np_gray_image.copy() - min_val) / (max_val - min_val)
    lbm_corrupted = [np_gray_image_min_max]

    for i in range(config.solver.max_fwd_steps):
        # niu_array = np.array(sum([[niu_array[i]]*dt_array[i] for i in range(len(niu_array))], []))
        # print(corrupt_sched[i])
        niu = np.array([niu_array[i]]*corrupt_sched[i])
        # print(niu)

        solver = LBM_ADE_Solver(
            (L, L),
            niu,
            niu,
            conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[i], dtype=np.float32),
            spectralTurbulenceGenerator
            )

        solver.init(np_gray_image_rot90.copy()) 
        solver.solve(iterations = dt_array[i])
        img = solver.rho
        blurred_by_lbm = np.rot90(img.to_numpy(), k=1) # torch to numpy + rotation
        min_val = blurred_by_lbm.min()
        max_val = blurred_by_lbm.max()
        lbm_corrupted.append((blurred_by_lbm - min_val) / (max_val - min_val))
        
    match_sim_numbers.plot_matrices_in_grid(lbm_corrupted, columns=4)
    match_sim_numbers.plot_matrix_side_by_side(lbm_corrupted[-1], blurred_by_dct_all[-1], title="LBM BLURR")
    match_sim_numbers.plot_matrix(blurred_by_dct_all[-1]-lbm_corrupted[-1], title="Difference DCT schedule - LBM")
    print(f'Difference: {np.mean((blurred_by_dct_all[-1]-lbm_corrupted[-1])**2)}')
    # print(f'Maximal value of DCT blurr: {blurred_by_dct.max()}')
    # print(f'Maximal value of LBM blurr: {blurred_by_lbm.max()}')

if __name__ == '__main__':
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    main()