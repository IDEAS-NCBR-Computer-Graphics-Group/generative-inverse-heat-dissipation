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

from model_code.utils import DCTBlur

#%% ################ helper functions ################
@njit
def get_r_from_xy(x, y, x0=0., y0=0.):
    r = np.sqrt((x0 - x)**2 + (y0 - y)**2)
    return r

# @jit(nopython=True, parallel=True)
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
diffusivity0 = 8.
advection_coeff_0 = 0*16./64
tc = 4. # 1

L = 256 # 2 # domain size
n = 256 # discretization

# IC
np.random.seed(0)
N_circles = int(1e2)
intensity_min = 1.
intensity_max = 2.

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

def Gaussian_blur(image, time, diff_coeff):
  sigma = np.sqrt(2 * diff_coeff * time)
  blurred_img = gaussian_filter(image, sigma=sigma)
  return blurred_img



#%% ################ RUN ################

P1 = fft_ade(initial_condition, tc, diffusivity0, advection_coeff_0)
P2 = Gaussian_blur(initial_condition, tc, diffusivity0)

plot_matrix(P1, title="FFT Blurr")
plot_matrix(P2, title="Gaussian Blurr")
plot_matrix(P2-P1, title="Difference")

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

def Gaussian_blur_dct_match(image, time, diff_coeff):
  sigma = np.sqrt(2 * diff_coeff * time)
  blurred_img = gaussian_filter(image, sigma=sigma)
  return blurred_img

# %%
t_initial_condition = torch.from_numpy(initial_condition)
t_initial_condition = t_initial_condition.unsqueeze(0)

fwd_steps = 50* torch.ones(1, dtype=torch.long)

P3 = dctBlur(t_initial_condition, fwd_steps).float()
P3 = P3.squeeze().numpy()
plot_matrix(P3, title="DCT Blurr")


# %%
sigma = model.blur_schedule[fwd_steps]  
P4 = gaussian_filter(initial_condition, sigma=sigma)
plot_matrix(P4, title="Sigma Blurr")

plot_matrix(P4-P3, title="Final Difference")
# corruption_amount = np.sqrt(sigma**2)

# %%
