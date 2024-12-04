#%% import stufff
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import taichi as ti
from model_code.utils import DCTBlur
from numerical_solvers.solvers.LBM_ADE_Solver import LBM_ADE_Solver
from tests.configs.ffhq_128_lbm_ade_example import get_config
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from configs import conf_utils, match_sim_numbers
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from numerical_solvers.visualization import plotting

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

def map01(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)

def calc_diff(x, y):
    return np.sqrt(x**2 - y**2)

def main():
    config = get_config()

    config.model.K = 3
    config.model.blur_sigma_max = 128
    config.model.blur_sigma_min = 0.5
    config.model.blur_schedule = np.exp(np.linspace(
        np.log(config.model.blur_sigma_min),
        np.log(config.model.blur_sigma_max),
        config.model.K
        ))
    config.model.blur_schedule = np.array([0] + list(config.model.blur_schedule))

    bs = config.model.blur_schedule
    bs_diff = [calc_diff(bs[i], bs[i-1]) for  i in range(1,len(bs))]
    bs_diff = [bs[0] , *bs_diff]
    dctBlur = DCTBlur(config.model.blur_schedule, image_size=config.data.image_size, device="cpu")
    dctBlur_diff = DCTBlur(bs_diff, image_size=config.data.image_size, device="cpu")

    print(bs)
    print(bs_diff)

    L = config.data.image_size
    np_gray_image = load_image(config, L)
    # np_gray_image_rot90 = np.rot90(np_gray_image.copy(), k=-1)

    plotting.plot_matrix(map01(np_gray_image), title="IC")
    t_init = torch.from_numpy(np_gray_image).unsqueeze(0)

    fwd_steps = config.model.K * torch.ones(1, dtype=torch.long)
    blurred_by_dct = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
    plotting.plot_matrix(map01(blurred_by_dct), title=f'DCT schedule Blurr for sigma')

    blurred_by_dct_all = []
    blurred_by_delta_dct_all = []
    blurred_by_dct_m1 = np_gray_image.copy()

    for i in range(len(config.model.blur_schedule)):
        fwd_steps = i * torch.ones(1, dtype=torch.long)
        blurred_by_dct = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
        blurred_by_dct_all.append(map01(blurred_by_dct))

        blurred_by_dct_diff = dctBlur_diff(torch.Tensor(blurred_by_dct_m1.copy()).unsqueeze(0), fwd_steps).float().squeeze().numpy()
        blurred_by_delta_dct_all.append(map01(blurred_by_dct_diff))
        blurred_by_dct_m1 = blurred_by_dct_diff

    plotting.plot_matrices_in_grid(blurred_by_dct_all, columns=4)
    plotting.plot_matrices_in_grid(blurred_by_delta_dct_all, columns=4)

    dFo_array = match_sim_numbers.calc_Fo(bs_diff, L)

    niu_min = 1E-4 * 1/6
    niu_max = 1./6

    dt_array, niu_array, *_ = match_sim_numbers.calculate_t_niu_array_from_0(
        dFo_array, niu_min, niu_max, L)

    corrupt_sched =  np.array(list(np.cumsum(dt_array)), dtype=int)
    print(f'niu_array = {niu_array}')
    print(f'sigma = {config.model.blur_schedule}')
    print(f'Fo={dFo_array}')
    print(f'dt_array = {dt_array}')
    print(f'corrupt_sched = {corrupt_sched}')

    config.solver.max_fwd_steps = config.model.K
    config.solver.n_denoising_steps = config.model.K - 1
    config.solver.corrupt_sched = corrupt_sched

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32)
    config.turbulence.turb_intensity = conf_utils.lin_schedule(0, 0, corrupt_sched[-1], dtype=np.float32)
    config.solver.niu = config.solver_bulk_visc = np.array(sum([[niu_array[i]]*dt_array[i] for i in range(len(niu_array))], []))
    config.solver.final_lbm_step = int(corrupt_sched[-1])

    np_gray_image_rot90 = np.rot90(np_gray_image.copy(), k=-1)

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

    print(config.solver.niu)
    solver = LBM_ADE_Solver(
        (L, L),
        config.solver.niu,
        config.solver.niu,
        conf_utils.lin_schedule(1./3, 1./3, config.solver.final_lbm_step, dtype=np.float32),
        spectralTurbulenceGenerator
        )

    solver.init(np_gray_image_rot90.copy()) 
    lbm_corrupted = []
    for lbm_iter in dt_array:
        solver.solve(iterations=lbm_iter)
        img = solver.rho
        blurred_by_lbm = np.rot90(img.to_numpy(), k=1)
        lbm_corrupted.append(map01(blurred_by_lbm))

    plotting.plot_matrices_in_grid(lbm_corrupted, columns=4)
    plotting.plot_matrix_side_by_side(lbm_corrupted[-1], blurred_by_dct_all[-1], title="LBM BLURR")
    plotting.plot_matrix(blurred_by_dct_all[-1]-lbm_corrupted[-1], title="Difference DCT schedule - LBM", range=(0,1))
    print(f'Difference: {np.mean((blurred_by_dct_all[-1]-lbm_corrupted[-1])**2)}')

if __name__ == '__main__':
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    main()
