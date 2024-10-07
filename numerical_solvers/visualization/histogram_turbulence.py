import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
import cv2

from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def compute_kolmogorov_spectrum(u, v, Lx, Ly):
    """
    Compute and plot the Kolmogorov spectrum using the straightforward energy spectrum calculation.

    Args:
        u: 2D array of x-velocity fluctuations.
        v: 2D array of y-velocity fluctuations.
        Lx: Length of the domain in the x-direction.
        Ly: Length of the domain in the y-direction.

    Returns:
        k: 1D array of wavenumbers.
        energy_spectrum: 1D array of energy spectrum values.
    """
    # Grid dimensions
    Nx, Ny = u.shape

    # Compute the 2D Fourier transforms of the velocity fields
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    # Compute the energy spectrum as the sum of the squared magnitudes of the Fourier coefficients
    energy_spectrum = np.abs(u_hat)**2 + np.abs(v_hat)**2

    # Average the energy spectrum over one dimension (e.g., the y-dimension)
    energy_spectrum = np.mean(energy_spectrum, axis=0)

    # Corresponding wavenumbers
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    k = np.sqrt(kx**2 + ky**2)

    # Sort the spectrum by wavenumber
    idx = np.argsort(k)
    k_sorted = k[idx]
    energy_spectrum_sorted = energy_spectrum[idx]

    return k_sorted, energy_spectrum_sorted

def plot_v_component_distribution(v_data, title):
    # Ensure the data is a 1D array
    v_data = v_data.reshape(-1)  # Flatten the data

    num_bins = 128
    counts, bins = np.histogram(v_data, bins=num_bins, density=True)

    bin_widths = np.diff(bins)
    integral = np.sum(counts * bin_widths)

    # Fit Gaussian distribution to the data
    mu, sigma = norm.fit(v_data)
    
    # Create an array over the full range specified
    x_range = np.linspace(-max(v_data.max(), -v_data.min()) * 2, max(v_data.max(), -v_data.min()) * 2, 256)
    gaussian_fit = norm.pdf(x_range, mu, sigma)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted size for normal plot

    # Plot the histogram and Gaussian fit
    ax.hist(v_data, bins=num_bins, density=True, alpha=0.6, color='g')
    ax.set_xlim([-max(v_data.max(), -v_data.min()), max(v_data.max(), -v_data.min())])
    # ax.set_xlim(-1E-2, 1E-2)
    # ax.set_ylim(0, 1000)
    ax.plot(x_range, gaussian_fit, 'r-', lw=2)
    
    # Set plot titles and labels
    ax.set_title(f'{title}', fontsize=10)
    ax.set_xlabel('v component')
    ax.set_ylabel('Probability Density')

    # Return the figure object
    return fig, mu, sigma**2, integral # Return mean, variance along with the plot

# Define domain and grid parameters
domain_size = (1.0, 1.0)  # Size of the domain in meters (Lx, Ly)
grid_size = (256, 256)     # Number of grid points (Nx, Ny)



# Initialize the SpectralTurbulenceGenerator (assuming it is already defined somewhere)
turbulence_generator = SpectralTurbulenceGenerator(
    domain_size, grid_size, turb_intensity=0.0001, noise_limiter=(-1E-3, 1E-3), energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0)), frequency_range= {'k_min': 2.0 * np.pi / min(domain_size), 'k_max': 2.0 * np.pi / (min(domain_size) / 1024)}
)

# Generate the turbulent velocity field
u, v = turbulence_generator.generate_turbulence(time=0.5)
# Nx, Ny = grid_size
# mean=0.0
# std_dev=1.0
# u = np.random.normal(mean, std_dev, size=(Ny, Nx))
# v = np.random.normal(mean, std_dev, size=(Ny, Nx))

k, energy_spectrum = compute_kolmogorov_spectrum(u, v, 1, 1)

fig = plt.figure(figsize=(6, 4))
        
# Create Axes with space for the title and labels
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])  # [left, bottom, width, height] as fractions of the figure size
# Plot the current energy spectrum
ax.set_ylim([1E-7, 1E2])
ax.loglog(k[1:len(k) // 2], energy_spectrum[1:len(k) // 2], 'b>', label='Energy spectrum')


# Add grid and labels
ax.grid(True, which="both", ls="--")
ax.set_xlabel(r"Wavenumber $k$")
ax.set_ylabel(r"Energy Spectrum $E(k)$")

u = u.flatten()
v = v.flatten()

# Plot and obtain statistics for the v-component distribution
v_fig, v_mean, v_variance, v_integral = plot_v_component_distribution(v, "v-component distribution")

# Plot and obtain statistics for the u-component distribution
u_fig, u_mean, u_variance, u_integral = plot_v_component_distribution(u, "u-component distribution")

# Save the plots
fig.savefig("spectrum.png")
v_fig.savefig("v_distribution.png")
u_fig.savefig("u_distribution.png")  

print("V-component Mean:", v_mean)
print("V-component Variance:", v_variance)
print("V-component Integral:", v_integral)


print("U-component Mean:", u_mean)
print("U-component Variance:", u_variance)
print("U-component Integral:", u_integral)


