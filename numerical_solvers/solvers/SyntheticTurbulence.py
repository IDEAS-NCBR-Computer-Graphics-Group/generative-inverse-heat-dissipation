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


class SpectralTurbulenceGenerator:
    def __init__(self, domain_size, grid_size, 
                 turb_intensity, 
                 noise_limiter = (-1E-3,1E-3),
                 energy_spectrum=None, 
                 frequency_range=None, 
                 dt_turb=1E-4, 
                 is_div_free = False):
        """
        Initialize the TurbulenceGenerator with domain and grid parameters.
        
        Parameters:
        - domain_size: tuple (Lx, Ly) representing the size of the domain
        - grid_size: tuple (Nx, Ny) representing the number of grid points in each direction
        - turb_intensity: float, turbulence intensity scaling factor
        - num_modes: int, number of random Fourier modes to generate (used in RFM method)
        - energy_spectrum: function, energy spectrum function (optional)
        """
        self.Lx, self.Ly = domain_size
        self.Nx, self.Ny = grid_size
        self.desired_std = 1. # desired standard deviation of the output 
        self.turb_intensity = turb_intensity
        self.energy_spectrum = energy_spectrum if energy_spectrum else self.default_energy_spectrum
        self.frequency_range = frequency_range if frequency_range else {'k_min': 2.0 * np.pi / min(domain_size), 'k_max': 2.0 * np.pi / (min(domain_size) / 20)}
        
        # Fourier transform wave numbers
        self.kx = fft.fftfreq(self.Nx, d=self.Lx/self.Nx) * 2 * np.pi
        self.ky = fft.fftfreq(self.Ny, d=self.Ly/self.Ny) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K = np.sqrt(self.KX**2 + self.KY**2)

        # Initialize the phases once and use them in each run
        self.phase_u = np.random.rand(self.Nx, self.Ny) * 2 * np.pi
        self.phase_v = np.random.rand(self.Nx, self.Ny) * 2 * np.pi
        
        self.amplitude = np.where(self.K != 0, self.energy_spectrum(self.K), 0)
        self.amplitude = np.where((self.K >= self.frequency_range['k_min']) & (self.K <= self.frequency_range['k_max']), self.amplitude, 0.0)

        self.dt_turb = dt_turb
        self.omega = self.dt_turb*np.sqrt(self.KX**2 + self.KY**2)
        
        self.noise_limiter = noise_limiter
        self.is_div_free = is_div_free
        

    def default_energy_spectrum(self, k):
        """
        Default energy spectrum function based on Kolmogorov's -5/3 law.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            spectrum = k ** (-5.0 / 3.0)
            spectrum[np.isinf(spectrum)] = 0  # Replace any infinities (from divide by zero) with 0
        return spectrum

    def tanh_limiter(self, x, min_val, max_val, sharpness=1.0):
        """
        Applies a tanh-like limiter to smoothly constrain x within [min_val, max_val] with adjustable sharpness.
        
        Args:
        - x: Input value or array.
        - min_val: Minimum allowable value.
        - max_val: Maximum allowable value.
        - sharpness: Controls the sharpness of the transition; higher values make the transition sharper.
        
        Returns:
        - Limited value(s) of x within the range [min_val, max_val].
        """
        mid_val = (max_val + min_val) / 2
        range_val = (max_val - min_val) / 2
        # Adjusted tanh with sharpness
        return mid_val + range_val * np.tanh(sharpness * (x - mid_val) / range_val)

    def limit_velocity_field(self, u, v, min_val, max_val):
        """
        Limits the magnitude of a velocity field with u, v components using a tanh-like limiter.
        
        Args:
        - u: 2D numpy array for the u component of the velocity.
        - v: 2D numpy array for the v component of the velocity.
        - min_val: Minimum allowable magnitude of the velocity.
        - max_val: Maximum allowable magnitude of the velocity.
        
        Returns:
        - u_limited: Limited u component of the velocity field.
        - v_limited: Limited v component of the velocity field.
        """
        # Calculate the magnitude of the velocity field
        velocity_magnitude = np.sqrt(u**2 + v**2)
        
        # Apply the tanh limiter to the velocity magnitudes
        limited_magnitude = self.tanh_limiter(velocity_magnitude, min_val, max_val)
        
        # Avoid division by zero; use a small factor if magnitude is less than 1E-6
        small_factor = 1E-9
        direction_factor = np.where(velocity_magnitude < small_factor, small_factor, limited_magnitude / velocity_magnitude)
        
        # Adjust u and v components to match the new limited magnitude while preserving direction
        upscale = 1. #1E1
        direction_factor *=upscale
        
        u_limited = u * direction_factor
        v_limited = v * direction_factor

        return u_limited, v_limited

    def generate_turbulence(self, time):
        """
        Generates 2D synthetic turbulence using a spectral method at a specific time.
        
        Parameters:
        - time: float, specific time at which to generate the turbulence
        
        Returns:
        - u: 2D array of x-velocity fluctuations (Ny, Nx)
        - v: 2D array of y-velocity fluctuations (Ny, Nx)
        """

        u_hat = self.turb_intensity * self.amplitude * np.exp(1j * (self.phase_u + self.omega * time))
        v_hat = self.turb_intensity * self.amplitude * np.exp(1j * (self.phase_v + self.omega * time))


        if self.is_div_free:
            # Compute k^2 = kx^2 + ky^2
            k2 = self.KX**2 + self.KY**2
            k2[0, 0] = 1.0  # Avoid division by zero at the zero frequency component
            
            # Compute the divergence-free components
            divergence_factor = (u_hat * self.KX + v_hat * self.KY) / k2
            
            u_hat_div_free = u_hat - divergence_factor * self.KX
            v_hat_div_free = v_hat - divergence_factor * self.KY

            # Set the zero frequency component to zero
            u_hat_div_free[0, 0] = 0
            v_hat_div_free[0, 0] = 0
            
            u_hat = u_hat_div_free
            v_hat = v_hat_div_free
        
        
        u = np.real(fft.ifft2(u_hat))
        v = np.real(fft.ifft2(v_hat))

       
        if self.turb_intensity < 1E-14:
            u,v = 0*self.K, 0*self.K #avoid division by 0 in np.std(u)
        else:
            # u *= (self.desired_std/np.std(u))
            # v *= (self.desired_std/np.std(v))
            
            # todo: the followin would chagne the std deviation
            u *= self.turb_intensity / np.std(u)
            v *= self.turb_intensity / np.std(v)

        # Apply limiter
        min_noise, max_noise = self.noise_limiter

        # Limiting the values of u and v elementwise
        # u = np.clip(u, min_noise, max_noise)
        # v = np.clip(v, min_noise, max_noise)
    
        u, v = self.limit_velocity_field(u, v, min_noise, max_noise)
        
        return np.float32(u), np.float32(v)


class RFMTurbulenceGenerator:
    def __init__(self, domain_size, grid_size, turb_intensity, num_modes, energy_spectrum=None):
        """
        Initialize the TurbulenceGenerator with domain and grid parameters.
        
        Parameters:
        - domain_size: tuple (Lx, Ly) representing the size of the domain
        - grid_size: tuple (Nx, Ny) representing the number of grid points in each direction
        - turb_intensity: float, turbulence intensity scaling factor
        - num_modes: int, number of random Fourier modes to generate (used in RFM method)
        - energy_spectrum: function, energy spectrum function (optional)
        """
        self.Lx, self.Ly = domain_size
        self.Nx, self.Ny = grid_size
        self.turb_intensity = turb_intensity
        self.num_modes = num_modes
        self.energy_spectrum = energy_spectrum if energy_spectrum else self.default_energy_spectrum

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize mode properties
        self.kx = np.random.uniform(-np.pi / self.dx, np.pi / self.dx, self.num_modes)
        self.ky = np.random.uniform(-np.pi / self.dy, np.pi / self.dy, self.num_modes)
        self.phase = np.random.uniform(0, 2 * np.pi, self.num_modes)
        self.theta = np.random.uniform(0, 2 * np.pi, self.num_modes)

        # Calculate other mode properties based on the above
        self.k = np.sqrt(self.kx ** 2 + self.ky ** 2)
        self.amplitude = self.turb_intensity * np.sqrt(2 * self.energy_spectrum(self.k))
        self.omega = np.sqrt(self.kx**2 + self.ky**2)

    def default_energy_spectrum(self, k):
        """
        Default energy spectrum function based on Kolmogorov's -5/3 law.
        """
        return k ** (-5.0 / 3.0)

    def generate_turbulence(self, time):
        """
        Generate a 2D synthetic turbulence field at a specific time using the Random Fourier Modes (RFM) method.
        
        Parameters:
        - time: float, specific time at which to generate the turbulence
        
        Returns:
        - u: 2D array of synthetic turbulence in the x-direction (Ny, Nx)
        - v: 2D array of synthetic turbulence in the y-direction (Ny, Nx)
        """
        # u = np.zeros((self.Ny, self.Nx))
        # v = np.zeros((self.Ny, self.Nx))

        # for i in range(self.num_modes):
        #     u_mode = self.amplitude[i] * np.cos(self.kx[i] * self.X + self.ky[i] * self.Y + self.omega[i] * time + self.phase[i]) * np.cos(self.theta[i])
        #     v_mode = self.amplitude[i] * np.sin(self.kx[i] * self.X + self.ky[i] * self.Y + self.omega[i] * time + self.phase[i]) * np.sin(self.theta[i])

        #     u += u_mode
        #     v += v_mode

              # Vectorized approach: compute all modes at once
        kxX = self.kx[:, np.newaxis, np.newaxis] * self.X[np.newaxis, :, :]
        kyY = self.ky[:, np.newaxis, np.newaxis] * self.Y[np.newaxis, :, :]
        omega_t_phase = self.omega[:, np.newaxis, np.newaxis] * time + self.phase[:, np.newaxis, np.newaxis]
        
        u_modes = self.amplitude[:, np.newaxis, np.newaxis] * np.cos(kxX + kyY + omega_t_phase) * np.cos(self.theta[:, np.newaxis, np.newaxis])
        v_modes = self.amplitude[:, np.newaxis, np.newaxis] * np.sin(kxX + kyY + omega_t_phase) * np.sin(self.theta[:, np.newaxis, np.newaxis])
        
        # Summing over the first axis to combine contributions from all modes
        u = np.sum(u_modes, axis=0)
        v = np.sum(v_modes, axis=0)
        
        return u, v