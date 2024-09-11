import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import taichi as ti
from math import pi

class RFMTurbulenceGenerator:
    def __init__(self, domain_size, grid_size, turb_intensity, num_modes, energy_spectrum=None):
        """
        Initialize the TurbulenceGenerator with domain and grid parameters.
        RFM = Random Fourier Modes
        
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