import torch as t

class SpectralTurbulenceGenerator(t.nn.Module):
    def __init__(
            self,
            domain_size,
            grid_size, 
            std_dev, 
            noise_limiter = (-1E-3,1E-3),
            energy_spectrum=None, 
            frequency_range=None, 
            dt_turb=1E-4, 
            is_div_free = False,
            device='cuda'
            ):
        """
        Initialize the TurbulenceGenerator with domain and grid parameters.
        
        Parameters:
        - domain_size: tuple (Lx, Ly) representing the size of the domain
        - grid_size: tuple (Nx, Ny) representing the number of grid points in each direction
        - std_dev: float, turbulence intensity scaling factor - standard deviation scaling factor
        - num_modes: int, number of random Fourier modes to generate (used in RFM method)
        - energy_spectrum: function, energy spectrum function (optional)
        """
        self.device = device
        self.Lx, self.Ly = domain_size
        self.Nx, self.Ny = grid_size
        self.desired_std = 1. # desired standard deviation of the output 
        self.std_dev = std_dev
        self.energy_spectrum = energy_spectrum if energy_spectrum else self.default_energy_spectrum.to(device)
        self.frequency_range = frequency_range if frequency_range else {'k_min': 2.0 * t.pi / min(domain_size), 'k_max': 2.0 * t.pi / (min(domain_size) / 20)}


        # Fourier transform wave numbers
        self.kx = (t.fft.fftfreq(self.Nx, d=self.Lx/self.Nx) * 2 * t.pi).to(device)
        self.ky = (t.fft.fftfreq(self.Ny, d=self.Ly/self.Ny) * 2 * t.pi).to(device)
        self.KX, self.KY = t.meshgrid(self.kx, self.ky)
        self.K = t.sqrt(self.KX**2 + self.KY**2).to(device)

        # Initialize the phases once and use them in each run
        self.phase_u = (t.rand(self.Nx, self.Ny) * 2 * t.pi).to(device)
        self.phase_v = (t.rand(self.Nx, self.Ny) * 2 * t.pi).to(device)
        
        self.amplitude = (t.where(self.K != 0, (self.energy_spectrum(self.K)), 0)).to(device)
        self.amplitude = (t.where((self.K >= self.frequency_range['k_min']) & (self.K <= self.frequency_range['k_max']), self.amplitude, 0.0)).to(device)

        self.dt_turb = dt_turb
        self.omega = self.dt_turb*t.sqrt(self.KX**2 + self.KY**2)
        
        self.noise_limiter = noise_limiter
        self.is_div_free = is_div_free
        
    def randomize(self):
        # Initialize the phases once and use them in each run
        self.phase_u = (t.rand(self.Nx, self.Ny) * 2 * t.pi).to(self.device)
        self.phase_v = (t.rand(self.Nx, self.Ny) * 2 * t.pi).to(self.device)

    def default_energy_spectrum(self, k):
        """
        Default energy spectrum function based on Kolmogorov's -5/3 law.
        """
        with t.errstate(divide='ignore', invalid='ignore'):
            spectrum = k ** (-5.0 / 3.0)
            spectrum[t.isinf(spectrum)] = 0  # Replace any infinities (from divide by zero) with 0
        return spectrum

    def tanh_limiter(self, x, min_val, max_val, sharpness=1.0):
        """
        Applies a tanh-like limiter to smoothly constrain x within [min_val, max_val] with adjustable sharpness.
        
        Args:
        - x: Itut value or array.
        - min_val: Minimum allowable value.
        - max_val: Maximum allowable value.
        - sharpness: Controls the sharpness of the transition; higher values make the transition sharper.
        
        Returns:
        - Limited value(s) of x within the range [min_val, max_val].
        """
        mid_val = (max_val + min_val) / 2
        range_val = (max_val - min_val) / 2
        # Adjusted tanh with sharpness
        return mid_val + range_val * t.tanh(sharpness * (x - mid_val) / range_val)

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
        velocity_magnitude = t.sqrt(u**2 + v**2)
        
        # Apply the tanh limiter to the velocity magnitudes
        limited_magnitude = self.tanh_limiter(velocity_magnitude, min_val, max_val)
        
        # Avoid division by zero; use a small factor if magnitude is less than 1E-6
        small_factor = 1E-9
        direction_factor = t.where(velocity_magnitude < small_factor, small_factor, limited_magnitude / velocity_magnitude)
        
        # Adjust u and v components to match the new limited magnitude while preserving direction
        upscale = 1. #1E1
        direction_factor *=upscale
        
        u_limited = u * direction_factor
        v_limited = v * direction_factor

        return u_limited, v_limited

    def generate_turbulence(self, time: int):
        """
        Generates 2D synthetic turbulence using a spectral method at a specific time.
        
        Parameters:
        - time: int, specific time step at which to generate the turbulence
        
        Returns:
        - u: 2D array of x-velocity fluctuations (Ny, Nx)
        - v: 2D array of y-velocity fluctuations (Ny, Nx)
        """

        u_hat = self.amplitude * t.exp(1j * (self.phase_u + self.omega * time))
        v_hat = self.amplitude * t.exp(1j * (self.phase_v + self.omega * time))


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
        
        
        u = t.real(t.fft.ifft2(u_hat))
        v = t.real(t.fft.ifft2(v_hat))

       
        if self.std_dev< 1E-14:
            u,v = 0*self.K, 0*self.K #avoid division by 0 in t.std(u)
        else:
            # u *= (self.desired_std/t.std(u))
            # v *= (self.desired_std/t.std(v))
            
            # # Normalize u and v to have a standard deviation of 1
            # u /= t.std(u)
            # v /= t.std(v)
            
            # # todo: would the followin chagne the std deviation?
            # u *= self.std_dev 
            # v *= self.std_dev 
            u *= self.std_dev/ t.std(u)
            v *= self.std_dev / t.std(v)
            

        # Apply limiter
        min_noise, max_noise = self.noise_limiter

        # Limiting the values of u and v elementwise
        # u = t.clip(u, min_noise, max_noise)
        # v = t.clip(v, min_noise, max_noise)
    
        u, v = self.limit_velocity_field(u, v, min_noise, max_noise)
        
        return u.float(), v.float()  # equivalent of self.to(torch.float32)

