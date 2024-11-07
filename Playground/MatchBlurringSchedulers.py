# %% ################ imports ################

import os, sys
import numpy as np
from numba import jit, njit
from numpy.testing import assert_almost_equal
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
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
from configs import conf_utils
from configs.mnist.small_mnist_lbm_ade_turb_config import get_config
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter

ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)

# %% ################ helper functions ################

@njit
def get_r_from_xy(x, y, x0=0., y0=0.):
    r = np.sqrt((x0 - x)**2 + (y0 - y)**2)
    return r

#same as @jit(nopython=True, parallel=True)
@njit
def make_circle(xx, yy, initial_condition, r0=0.25, x0=0., y0=0., intensity=1.):
    for i in range(n):
        for j in range(n):
            r = get_r_from_xy(xx[i,j], yy[i,j], x0, y0)
            if r < r0:
                initial_condition[i,j] = intensity

def plot_matrix(matrix, title="Temperature Map (Matrix)"):
  # Create a heatmap using matplotlib
  plt.imshow(matrix, cmap='hot', interpolation='nearest')
  plt.colorbar()
  plt.title(title)
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.show()
  
# %% ################ SETUP ################
diffusivity0 = 20. # 8.
advection_coeff_0 = 0*16./64
tc = 10. # 4 #  1
L = 128 # 2 # domain size
n = 128 # discretization

# IC
np.random.seed(0)
N_circles = int(1e2)
intensity_min = 0.
intensity_max = 1.

# no of samples
num_matrices = 1 # 100
tensor_size = (num_matrices, n, n)

# %% ################ IC by random circles ################
cirles = np.array([np.random.uniform(0,L/10.,N_circles),
                  np.random.uniform(0, L, N_circles),
                  np.random.uniform(0, L, N_circles),
                  np.random.uniform(intensity_min, intensity_max, N_circles)]).T

initial_condition = np.zeros((n, n))
x = np.linspace(0, L, n, endpoint=True)
y = np.linspace(0, L, n, endpoint=True)
xx, yy = np.meshgrid(x, y)

for r0, x0, y0, intensity0 in cirles:
  make_circle(xx, yy, initial_condition, r0, x0, y0, intensity0)
plot_matrix(initial_condition, title="IC")
np_init_gray_image = np.rot90(initial_condition.copy(), k=-1)
t_initial_condition = torch.from_numpy(initial_condition).unsqueeze(0)

# %% ################ Solvers ################
def fft_ade(image, time, diff_coeff, advect_coeff):
    n = image.shape[0]
    L = n  # Assuming square image, and length L = n
    F = fft2(image)

    # Solve the heat equation in the Fourier space
    x2 = np.array([float(i) if i < n/2. else float(-(n-i)) for i in range(0,n)])
    k2, k1 = np.meshgrid(x2, x2)

    k1 *= 2.*np.pi/L
    k2 *= 2.*np.pi/L
    shift = np.exp(-complex(0,1)* advect_coeff* time * (k1 + k2))
    decay = np.exp(-diff_coeff * time* (k1**2 + k2**2))

    yinv = ifft2(np.multiply(F,shift*decay))
    P = np.real(yinv)
    return P

def calc_sigma(time, diff_coeff):
    sigma = np.sqrt(2 * diff_coeff * time)
    return sigma

class GaussianBlurLayerNaive(nn.Module):
    def __init__(self, blur_sigmas, device):
        super(GaussianBlurLayerNaive, self).__init__()
        # print(blur_sigmas)
        self.device = device
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)

    def forward(self, x, fwd_steps):
        sigmas = self.blur_sigmas[fwd_steps]
        
        for i in range(x.shape[0]):
            npx = x[i].numpy()
            sigma = float(sigmas[i].numpy())
            blurred_x = gaussian_filter(npx, sigma, mode="wrap")
            x[i] = torch.tensor(blurred_x).to(self.device)
        return x

# %% ################ RUN naive FFT ################

blurred_by_fft = fft_ade(initial_condition, tc, diffusivity0, advection_coeff_0)
sigma = calc_sigma(tc, diffusivity0)
print(f"Blurring sigma = {sigma}")

plot_matrix(blurred_by_fft, title="FFT Blurr")

# %% ################ Blurr by gaussian, show difference to FFT ################

blurred_by_gaussian = gaussian_filter(initial_condition, sigma=sigma, mode='wrap')
plot_matrix(blurred_by_gaussian, title="Gaussian Blurr")
plot_matrix(blurred_by_fft-blurred_by_gaussian, title="Difference (FFT-Gaussian)")


# %% ################ Import DCT ################

#%% ################ Import DCT ################
config = get_config()
model = config.model

#manual hack - as in small mnist
model.K = 50
model.blur_sigma_max = 20
model.blur_sigma_min = 0.5
model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                            np.log(model.blur_sigma_max), model.K))
model.blur_schedule = np.array(
    [0] + list(model.blur_schedule))  # Add the k=0 timestep


dctBlur = DCTBlur(model.blur_schedule, image_size=n, device="cpu")

# %% ################ RUN DCT schedule ################

fwd_steps = model.K* torch.ones(1, dtype=torch.long) 
blurred_by_dct = dctBlur(t_initial_condition, fwd_steps).float()
blurred_by_dct = blurred_by_dct.squeeze().numpy()
plot_matrix(blurred_by_dct, title="DCT schedule Blurr")

# %% ################ DCT schedule vs Gaussian Schedule ################

gaussianBlur = GaussianBlurLayerNaive(model.blur_schedule, device="cpu")
blurred_by_gaussian_schedule = gaussianBlur(t_initial_condition, fwd_steps).float().squeeze().numpy()

# plot_matrix(blurred_by_gaussian_schedule, title="Gaussian sigmas Blurr")
# plot_matrix(blurred_by_dct-blurred_by_gaussian_schedule, title="Final Difference")
# print(f"Blurring sigma = {calc_sigma(tc, diffusivity0)}")
# print(f"model.blur_schedule[fwd_steps] = {model.blur_schedule[fwd_steps]}")


# %% ################ ADE_LBM VS DCT ################
# assumption - max diffusivity for LBM
niu = conf_utils.lin_schedule(1. / 6, 1. / 6, config.solver.final_lbm_step, dtype=np.float32)
niu0= niu[0]
#mnist
# blur_sigma_max = 20
# L = 28 

# #ffhq256
# blur_sigma_max = 128
# L = 256


# #ffhq128 - estimate
blur_sigma_max = 20
L = 128


def calc_Fo(sigma, L):
    Fo = sigma / (L*L)
    return Fo

Fo = calc_Fo(blur_sigma_max, L)
print(f"Fo = {Fo}")

def get_timesteps_from_sigma(diffusivity, sigma):
    # sigma = np.sqrt(2 * diffusivity * tc)
    tc = sigma*sigma/(2*diffusivity)
    return int(tc)
    
lbm_iter = get_timesteps_from_sigma(niu0, blur_sigma_max)
print(f"lbm_iter = {lbm_iter}")

def get_sigma_from_Fo(Fo, L):
    sigma = Fo * L*L
    return sigma


print(f"sigma check = {get_sigma_from_Fo(Fo, L)}")

def get_timesteps_from_Fo_niu_L(Fo, diffusivity, L):
    # sigma = np.sqrt(2 * diffusivity * tc)
    sigma = Fo * L*L
    tc = sigma*sigma/(2*diffusivity)
    return int(tc)

print(f"lbm_iter check = {get_timesteps_from_Fo_niu_L(Fo, niu0, L)}")
# %% run solver

spectralTurbulenceGenerator = SpectralTurbulenceGenerator(config.turbulence.domain_size, initial_condition.shape, 
        0 * config.turbulence.turb_intensity, config.turbulence.noise_limiter,
        energy_spectrum=config.turbulence.energy_spectrum, 
        frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
        dt_turb=config.turbulence.dt_turb, 
    is_div_free=False)
solver = LBM_ADE_Solver(
    initial_condition.shape,
    niu, config.solver.bulk_visc, config.solver.cs2,
    spectralTurbulenceGenerator
    )
solver.init(np_init_gray_image) 
solver.solve(iterations=lbm_iter)

img = solver.rho
rho_np = np.rot90(img.to_numpy().copy(), k=1) # torch to numpy + rotation

plot_matrix(rho_np, title="LBM BLURR")
plot_matrix(blurred_by_dct-rho_np, title="Difference DCT schedule - LBM")
print(f'Maximal value of DCT blurr: {blurred_by_dct.max()}')
print(f'Maximal value of LBM blurr: {rho_np.max()}')
# %%
