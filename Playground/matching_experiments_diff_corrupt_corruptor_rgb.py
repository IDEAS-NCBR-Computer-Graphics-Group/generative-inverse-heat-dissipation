#%% import stufff
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import taichi as ti
import cv2

from model_code.utils import DCTBlur
from numerical_solvers.corruptors.LBM_ADE_Corruptor import LBM_ADE_Corruptor
from configs.ffhq.default_lbm_ade_ffhq_128_config import get_config
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from configs import conf_utils, match_sim_numbers
from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from numerical_solvers.visualization import plotting

def load_image(config, L):
    img_path = './numerical_solvers/runners/cat_768x768.jpg'
    image = cv2.imread(img_path) 
    image = cv2.resize(image, (L, L))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = config.data.transform(image)
    image = np.array(image)
    return image

def map01(image):
    return image

def calc_diff(x, y):
    return np.sqrt(x**2 - y**2)

def main():
    config = get_config()

    config.model.K = 4
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

    L = config.data.image_size
    np_gray_image = load_image(config, L)
    # np_gray_image_rot90 = np.rot90(np_gray_image.copy(), k=-1)
    # print(np_gray_image.shape)

    plotting.plot_matrix(np.transpose(np_gray_image, (1,2,0)), title="IC")
    t_init = torch.from_numpy(np_gray_image).unsqueeze(0)

    fwd_steps = config.model.K * torch.ones(1, dtype=torch.long)
    blurred_by_dct = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
    plotting.plot_matrix(np.transpose((map01(blurred_by_dct)), (2,1,0)), title=f'DCT schedule Blurr for sigma')

    blurred_by_dct_all = []
    blurred_by_delta_dct_all = []
    blurred_by_dct_m1 = np_gray_image.copy()

    for i in range(len(config.model.blur_schedule)):
        fwd_steps = i * torch.ones(1, dtype=torch.long)
        blurred_by_dct = dctBlur(t_init, fwd_steps).float().squeeze().numpy()
        blurred_by_dct_all.append(np.transpose(map01(blurred_by_dct), (1,2,0)))

        blurred_by_dct_diff = dctBlur_diff(torch.Tensor(blurred_by_dct_m1.copy()).unsqueeze(0), fwd_steps).float().squeeze().numpy()
        blurred_by_delta_dct_all.append(np.transpose(map01(blurred_by_dct_diff), (1,2,0)))
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

    config.solver.max_fwd_steps = config.solver.n_denoising_steps = config.model.K
    config.solver.corrupt_sched = corrupt_sched

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32)
    config.turbulence.turb_intensity = conf_utils.lin_schedule(0, 0, corrupt_sched[-1], dtype=np.float32)
    config.solver.niu = config.solver_bulk_visc = np.array(sum([[niu_array[i]]*dt_array[i] for i in range(len(niu_array))], []))
    config.solver.final_lbm_step = int(corrupt_sched[-1])

    corruptor = LBM_ADE_Corruptor(config=config, transform=config.data.transform)
    _ = corruptor._corrupt(t_init.squeeze(0), config.solver.max_fwd_steps)
    lbm_corrupted = [(np.transpose(i.squeeze(0).numpy(), (1,2,0))) for i in corruptor.intermediate_samples]

    plotting.plot_matrices_in_grid(lbm_corrupted, columns=4)
    plotting.plot_matrix_side_by_side(lbm_corrupted[-1], blurred_by_dct_all[-1], title="LBM BLURR")
    plotting.plot_matrix(blurred_by_dct_all[-1]-lbm_corrupted[-1], title="Difference DCT schedule - LBM", range=(0,1))
    print(f'Difference: {np.mean((blurred_by_dct_all[-1]-lbm_corrupted[-1])**2)}')

if __name__ == '__main__':
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    main()
