import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import taichi as ti
from math import pi

@ti.data_oriented
class GaussianTurbulenceGenerator:
    def __init__(self, domain_size, grid_size, 
                 turb_intensity, mean, variance,
                 noise_limiter = (-1E-3,1E-3)):
    
        self.mean = mean
        self.variance = variance
        self.noise_limiter = noise_limiter
        self.turb_intensity = turb_intensity
        
        
    @ti.func
    def get_gaussian_noise(self, mean: float, variance: float)  -> ti.types.vector(2, float):
        noise = ti.Vector([0.0, 0.0])  # We need two values, as the Box-Muller gives two values

        # u1 = ti.random()
        # u2 = ti.random()
        
        # # Box-Muller transform
        # r = ti.sqrt(-2.0 * ti.log(u1))
        # theta = 2.0 * pi * u2
        
        # # Generate standard normal distribution (mean 0, variance 1)
        # z0 = r * ti.cos(theta)
        # z1 = r * ti.sin(theta)
        
        # # Adjust for desired mean and variance
        # std_dev = ti.sqrt(variance)
        # noise[0] = mean + std_dev * z0
        # noise[1] = mean + std_dev * z1
        
        
        # build in ti.randn() 
        # Generate a random float sampled from univariate standard normal (Gaussian) distribution of mean 0 and variance 1, using the Box-Muller transformation.
        noise[0] = self.turb_intensity * ti.randn()
        noise[1] = self.turb_intensity * ti.randn()
        
        # Apply limiter

        min_noise, max_noise = self.noise_limiter
        
        # max_noise = 1E-1
        # min_noise = -1E-1
        
        noise[0] = ti.min(ti.max(noise[0], min_noise), max_noise)
        noise[1] = ti.min(ti.max(noise[1], min_noise), max_noise)
        
        return noise


@ti.func
def get_gaussian_noise(mean: float, variance: float)  -> ti.types.vector(2, float):
    noise = ti.Vector([0.0, 0.0])  # We need two values, as the Box-Muller gives two values
    
    # build in ti.randn() 
    # Generate a random float sampled from univariate standard normal (Gaussian) distribution of mean 0 and variance 1, using the Box-Muller transformation.
    noise[0] = ti.randn()
    noise[1] = ti.randn()
    
    # Apply limiter
    
    max_noise = 1E-1
    min_noise = -1E-1
    
    noise[0] = ti.min(ti.max(noise[0], min_noise), max_noise)
    noise[1] = ti.min(ti.max(noise[1], min_noise), max_noise)
    
    return noise
    
def generate_random_velocity_field(domain_size, grid_size, mean=0.0, std_dev=1.0):
    """
    Generate 2D u and v velocity components by sampling from a normal distribution.

    Args:
        domain_size: tuple (Lx, Ly) representing the size of the domain (not used in this naive approach).
        grid_size: tuple (Nx, Ny) representing the number of grid points in each direction.
        mean: float, the mean of the normal distribution (default is 0.0).
        std_dev: float, the standard deviation of the normal distribution (default is 1.0).
        seed: int, optional random seed for reproducibility.

    Returns:
        u: 2D array of x-velocity fluctuations (Ny, Nx).
        v: 2D array of y-velocity fluctuations (Ny, Nx).
    """
    Nx, Ny = grid_size


    # Generate u and v velocity components by sampling from a normal distribution
    u = np.random.normal(mean, std_dev, size=(Ny, Nx))
    v = np.random.normal(mean, std_dev, size=(Ny, Nx))

    return u, v
