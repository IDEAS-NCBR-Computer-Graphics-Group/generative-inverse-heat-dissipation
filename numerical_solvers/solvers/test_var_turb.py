

#%%
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from scipy.stats import norm

from SpectralTurbulenceGenerator import SpectralTurbulenceGenerator 

#%%

# Parameters for the Spectral Turbulence Generator
domain_size = (10.0, 10.0)  # Example domain size
grid_size = (64, 64)  # Example grid size
turb_intensity = 2.0  # Example turbulence intensity, can be any value
noise_limiter = (-1E6, 1E6)
# Initialize the Spectral Turbulence Generator
turb_gen = SpectralTurbulenceGenerator(
    domain_size, grid_size, turb_intensity, noise_limiter)

# Generate turbulence at a specific time (e.g., time = 0)
u, v = turb_gen.generate_turbulence(time=0.0)

# Calculate the standard deviation of the generated fields
std_u = np.std(u)
std_v = np.std(v)

# Normalize u and v to have a standard deviation of 1
# u = u / std_u
# v = v / std_v

# Calculate variance of the normalized fields (should be close to 1)
# variance_u_normalized = np.var(u_normalized)
# variance_v_normalized = np.var(v_normalized)

print(f"std of normalized u: {np.std(u):.4f}")
print(f"std of normalized v: {np.std(v):.4f}")

# Plot histogram of the normalized u component
plt.hist(u.flatten(), bins=30, density=True, alpha=0.6, color='g', label='u component histogram (normalized)')

# Fit a Gaussian to the normalized data
mu, std = norm.fit(u.flatten())
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

# %%
