#%% ################ imports ################

import os, sys
import numpy as np
from numba import jit, njit
from numpy.testing import assert_almost_equal
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
import matplotlib.pyplot as plt
import time
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms import GaussianBlur
import torch.nn as nn
import math

from model_code.utils import DCTBlur

#%% ################ helper functions ################

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
  
  
#%% ################ SETUP ################
diffusivity0 = 20. # 8.
advection_coeff_0 = 0*16./64
tc = 10. # 4 #  1

L = 256 # 2 # domain size
n = 256 # discretization

# IC
np.random.seed(0)
N_circles = int(1e2)
intensity_min = 0.
intensity_max = 1.

# no of samples
num_matrices = 1 # 100
tensor_size = (num_matrices, n, n)

#%% ################ IC by random circles ################
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

#%% ################ Solvers ################
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

def Gaussian_blur(image, time, diff_coeff):
  sigma = calc_sigma(time, diff_coeff)
  blurred_img = gaussian_filter(image, sigma=sigma)
  return blurred_img


class GaussianBlurLayerNaive(nn.Module):
    def __init__(self, blur_sigmas, device):
        super(GaussianBlurLayerNaive, self).__init__()
        print(blur_sigmas)
        self.device = device
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        # self.blur_sigmas = blur_sigmas


    def forward(self, x, fwd_steps):
        sigmas = self.blur_sigmas[fwd_steps]
        
        for i in range(x.shape[0]):
            npx = x[i].numpy()
            sigma = float(sigmas[i].numpy())
            blurred_x = gaussian_filter(npx, sigma)
            x[i] = torch.tensor(blurred_x).to(self.device)
        return x

      
         
#%% ################ RUN naive FFT ################

blurred_by_fft = fft_ade(initial_condition, tc, diffusivity0, advection_coeff_0)
sigma = calc_sigma(tc, diffusivity0)
print(f"Blurring sigma = {sigma}")
blurred_by_gaussian = gaussian_filter(initial_condition, sigma=sigma)


plot_matrix(blurred_by_fft, title="FFT Blurr")
plot_matrix(blurred_by_gaussian, title="Gaussian Blurr")
plot_matrix(blurred_by_fft-blurred_by_gaussian, title="Difference")

#%% ################ Import DCT ################
from configs.mnist.small_mnist import get_config
config = get_config()
model = config.model

# model.K = 50 
# model.blur_sigma_max = 20
# model.blur_sigma_min = 0.5
# model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
#                                             np.log(model.blur_sigma_max), model.K))
# model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
# sigmas = model.blur_schedule
dctBlur = DCTBlur(model.blur_schedule, image_size=n, device="cpu")

# %% ################ RUN DCT ################

t_initial_condition = torch.from_numpy(initial_condition)
t_initial_condition = t_initial_condition.unsqueeze(0)
fwd_steps = model.K* torch.ones(1, dtype=torch.long)

blurred_by_dct = dctBlur(t_initial_condition, fwd_steps).float()
blurred_by_dct = blurred_by_dct.squeeze().numpy()
plot_matrix(blurred_by_dct, title="DCT Blurr")

gaussianBlur = GaussianBlurLayerNaive(model.blur_schedule, device="cpu")


blurred_by_gaussian_schedule_v0 = gaussian_filter(initial_condition, sigma=model.blur_schedule[fwd_steps])

blurred_by_gaussian_schedule = gaussianBlur(t_initial_condition, fwd_steps).float()
blurred_by_gaussian_schedule = blurred_by_gaussian_schedule.squeeze().numpy()

plot_matrix(blurred_by_gaussian_schedule, title="Sigma Blurr")
plot_matrix(blurred_by_dct-blurred_by_gaussian_schedule, title="Final Difference")

print(f"Blurring sigma = {calc_sigma(tc, diffusivity0)}")
print(f"model.blur_schedule[fwd_steps] = {model.blur_schedule[fwd_steps]}")

