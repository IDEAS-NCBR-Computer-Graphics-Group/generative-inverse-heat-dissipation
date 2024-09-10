
#%% imports
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(123)

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.solvers.RFMTurbulenceGenerator import RFMTurbulenceGenerator
from numerical_solvers.solvers.GaussianTurbulenceGenerator import generate_random_velocity_field
from numerical_solvers.visualization.visualize_synthetic_turbulence_models import plot_kolmogorov_spectrum, plot_velocity_components, compute_divergence
from numerical_solvers.visualization.KolmogorovSpectrumPlotter import compute_kolmogorov_spectrum

#%% Example usage:
domain_size = (1.0, 1.0)
grid_size = (128, 128)
noise_limiter = (-1E6, 1E6)
# grid_size = (768, 768)
time = 1.
# turb_intensity = 0.1
turb_intensity = 1E-4
num_modes = 64


rfmTurbulenceGenerator = RFMTurbulenceGenerator(domain_size, grid_size, turb_intensity, num_modes)

frequency_range = {'k_min': 2.0 * np.pi / min(domain_size), 'k_max': 2.0 * np.pi / (min(domain_size) / 2048)}
# energy_spectrum = lambda k: np.where(np.isinf(k), 0, k) 
energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
    domain_size, grid_size, turb_intensity, noise_limiter, energy_spectrum=energy_spectrum, frequency_range=frequency_range)


#%% # Generate synthetic turbulence using both methods

u_rfm, v_rfm = rfmTurbulenceGenerator.generate_turbulence(time)
u_spec, v_spec = spectralTurbulenceGenerator.generate_turbulence(time)

divergence = compute_divergence(u_spec, u_spec, dx=1, dy=1)
print(f"max divergence is: {np.max(np.abs(divergence))}")  # Should be close to zero

# Generate random velocity fields
u_random, v_random = generate_random_velocity_field(domain_size, grid_size, 0, 1)
u_random *=turb_intensity
v_random *=turb_intensity

# Plot the Kolmogorov spectrum for both methods
plot_kolmogorov_spectrum(u_random, v_random, u_rfm, v_rfm, u_spec, v_spec, domain_size)


# Plot the velocity components
plot_velocity_components(u_random, v_random, u_rfm, v_rfm, u_spec, v_spec, time)


#%% plot histogram


# Normalize u and v to have a standard deviation of 1
u_spec = u_spec / np.std(u_spec)

print(f"std of normalized u: {np.std(u_spec):.4f}")
# print(f"std of normalized v: {np.std(v_spec):.4f}")

# Plot histogram of the normalized u component
hist, bin_edges, _ = plt.hist(u_spec.flatten(), bins=30, density=True, alpha=0.6, color='g', label='u component histogram (normalized)')


# Integrate the area under the histogram using the trapezoidal rule
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
area_under_histogram = np.trapz(hist, bin_centers)

print(f"Area under the histogram of u: {area_under_histogram:.4f}")



# Fit a Gaussian to the normalized data
mu, std = norm.fit(u_spec.flatten())
print(f"mu, std: {mu:.4f}  {std:.4f}")

# Plot the Gaussian fit
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian fit')

plt.title('Histogram of Normalized u Component with Gaussian Fit')
plt.xlabel('u values')
plt.ylabel('Density')
plt.legend()
plt.show()

#%% # check how fast it change in time

# u_rfm, v_rfm = rfmTurbulenceGenerator.generate_turbulence(time+1)
dt=0.025
u_spec1, v_spec1 = spectralTurbulenceGenerator.generate_turbulence(time)
u_spec2, v_spec2 = spectralTurbulenceGenerator.generate_turbulence(time+dt)
u_spec3, v_spec3 = spectralTurbulenceGenerator.generate_turbulence(time+2*dt)

plot_velocity_components(u_spec1, v_spec1, u_spec2, v_spec2, u_spec3, v_spec3, time)
