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


class GaussianBlurLayer1(nn.Module):
    def __init__(self, blur_sigmas, device):
        super(GaussianBlurLayer1, self).__init__()
        self.device = device
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)

    def calculate_kernel_size(self, sigma):
        # Formula to calculate kernel size from sigma
        return int(2 * math.ceil(3 * sigma) + 1)
    
    def forward(self, x, fwd_steps):
        sigmas = self.blur_sigmas[fwd_steps]

        for i in range(x.shape[0]):
            sigma = float(sigmas[i].item())
            kernel_size = self.calculate_kernel_size(sigma)
            
            # Ensure the kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Apply Gaussian blur using torchvision.transforms.GaussianBlur
            blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            
            x[i] = blur(x[i])
        
        return x   
      


class GaussianBlurLayer2(nn.Module):
    def __init__(self, blur_sigmas, device):
        super(GaussianBlurLayer2, self).__init__()
        self.device = device
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)

    def calculate_gaussian_kernel(self, sigma, kernel_size):
        """Create a 2D Gaussian kernel."""
        # Create a 1D Gaussian kernel
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gaussian_1d /= gaussian_1d.sum()

        # Outer product to get a 2D Gaussian kernel
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
        gaussian_2d /= gaussian_2d.sum()
        return gaussian_2d

    def calculate_kernel_size(self, sigma):
        """Formula to calculate kernel size from sigma."""
        return int(2 * math.ceil(3 * sigma) + 1)
    
    def forward(self, x, fwd_steps):
        sigmas = self.blur_sigmas[fwd_steps]
        
        # Prepare kernel sizes and the maximum size in the batch
        kernel_sizes = [self.calculate_kernel_size(sigma.item()) for sigma in sigmas]
        max_kernel_size = max(kernel_sizes)
        
        # Ensure max_kernel_size is odd
        if max_kernel_size % 2 == 0:
            max_kernel_size += 1
        
        # Initialize a list of kernels for the batch
        batch_size, channels, height, width = x.size()
        gaussian_kernels = []
        
        # Generate the Gaussian kernel for each image
        for sigma, kernel_size in zip(sigmas, kernel_sizes):
            gaussian_kernel = self.calculate_gaussian_kernel(sigma.item(), kernel_size)
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(self.device)
            # Pad the kernel to the max size
            pad = (max_kernel_size - kernel_size) // 2
            gaussian_kernel = F.pad(gaussian_kernel, (pad, pad, pad, pad))
            gaussian_kernels.append(gaussian_kernel)
        
        # Stack the kernels to apply them to the batch
        gaussian_kernels = torch.stack(gaussian_kernels).expand(batch_size, channels, -1, -1)
        
        # Apply the kernels to the batch using grouped convolution (one kernel per image)
        blurred = F.conv2d(x, gaussian_kernels, padding=max_kernel_size // 2, groups=batch_size)

        return blurred
    


# %% ################ RUN GaussianBlur from Torch ################

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class GaussianTorchBlur(nn.Module):
    def __init__(self, blur_sigmas,  device):
        super(GaussianTorchBlur, self).__init__()
        print(blur_sigmas)
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)

        
        """
        from scipy--> gaussian_filter()
        
        truncate : float, optional
          Truncate the filter at this many standard deviations.
          Default is 4.0.
        radius : None or int or sequence of ints, optional
            Radius of the Gaussian kernel. The radius are given for each axis
            as a sequence, or as a single number, in which case it is equal
            for all axes. If specified, the size of the kernel along each axis
            will be ``2*radius + 1``, and `truncate` is ignored.
            Default is None.
        """
        # TODO: this does not work as expected
        truncate=4.0
        # sd = float(blur_sigmas)
        # self.radius = int(truncate * sd + 0.5)
        # make the radius of the filter equal to truncate standard deviations
        radius = truncate * blur_sigmas + 0.5
        radius = radius.astype(int)
        self.kernel_sizes = torch.tensor(2*radius + 1).to(device) 


    def forward(self, x, fwd_steps):
        # if len(x.shape) == 4:
        #     sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        # elif len(x.shape) == 3:
        #     sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        
        sigmas = self.blur_sigmas[fwd_steps]
        kernel_size = self.kernel_sizes[fwd_steps]
        # sigmas += 1e-6

        # sigmas = sigmas.squeeze().tolist()
        sigmas = sigmas.tolist()
        transforms = [T.GaussianBlur(kernel_size=(kernel_size[i], kernel_size[i]), sigma=sigmas[i]) 
                      for i in range(len(sigmas))]
        
        # transforms = []
        # for i in range(len(sigmas)):
        #   sig=sigmas[i]
        #   tran = T.GaussianBlur(kernel_size=(self.kernel_size[i], self.kernel_size[i]), sigma=sig)
        
        # transforms = [T.GaussianBlur(kernel_size=(3,3), sigma=2) for i in range(4)]
            
        for i in range(x.shape[0]):
            x[i] = transforms[i](x[i])

        return x
      
gaussianTorchBlur = GaussianTorchBlur(model.blur_schedule, device="cpu")

# blurred_by_GaussianTorch = gaussianBlur(t_initial_condition, fwd_steps).float()
# blurred_by_GaussianTorch = blurred_by_GaussianTorch.squeeze().numpy()
# plot_matrix(blurred_by_GaussianTorch, title="GaussianTorch Blurr")