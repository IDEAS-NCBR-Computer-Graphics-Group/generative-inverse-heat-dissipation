import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Assume you have the function compute_kolmogorov_spectrum(u, v, Lx, Ly) defined already

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

class SpectrumHeatmapPlotter:
    def __init__(self, buffer_size, domain_size=(1.0, 1.0)):
        """
        Initialize the heatmap plotter with storage for energy spectrums and iteration numbers.

        Args:
            buffer_size (int): Number of iterations to store for the heatmap.
            domain_size (tuple): Domain size as (Lx, Ly).
        """
        self.domain_size = domain_size
        self.buffer_size = buffer_size
        self.spectrums = []  # List to store energy spectrums for each iteration
        self.iterations = []  # List to store iteration numbers
        self.k_values = None  # To store common wavenumber array (k)

    def add_spectrum(self, u, v, iteration):
        """
        Compute the Kolmogorov spectrum for the given velocity fields and store it.

        Args:
            u (np.ndarray): Velocity field in the x-direction.
            v (np.ndarray): Velocity field in the y-direction.
            iteration (int): Current iteration number.
        """
        Lx, Ly = self.domain_size

        # Compute the energy spectrum for the current data
        k, energy_spectrum = compute_kolmogorov_spectrum(u, v, Lx, Ly)

        # Initialize k_values on the first run
        if self.k_values is None:
            self.k_values = k

        # Ensure that the spectrum has the same length as the wavenumber array
        energy_spectrum = np.interp(self.k_values, k, energy_spectrum)

        # Add the spectrum and iteration to the storage
        self.spectrums.append(energy_spectrum)
        self.iterations.append(iteration)

        # Maintain the buffer size limit
        if len(self.spectrums) > self.buffer_size:
            self.spectrums.pop(0)
            self.iterations.pop(0)

    def plot_heatmap(self):
        """
        Plot a heatmap of the energy spectrum over iterations.
        """
        # Convert the list of spectrums to a 2D numpy array
        spectrums_array = np.array(self.spectrums).T  # Transpose to have iterations on the x-axis

        # Create the heatmap plot
        plt.figure(figsize=(1, 1))
        plt.title("Energy Spectrum Heatmap over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Wavenumber $k$")
        plt.imshow(
            spectrums_array,
            aspect='auto',
            extent=[min(self.iterations), max(self.iterations), min(self.k_values), max(self.k_values)],
            origin='lower',
            norm=LogNorm(),  # Use log scale for better visualization
            cmap='inferno'
        )
        plt.colorbar(label='Energy Spectrum $E(k)$')
        plt.show()

# Example usage
buffer_size = 50  # Store up to 50 iterations
domain_size = (1.0, 1.0)  # Example domain size

# Create the heatmap plotter
heatmap_plotter = SpectrumHeatmapPlotter(buffer_size, domain_size)

# Simulate adding spectrums over iterations
for iteration in range(100):
    # Simulate random velocity fields for example purposes
    u = np.random.rand(256, 256)
    v = np.random.rand(256, 256)
    
    # Add spectrum to heatmap plotter
    heatmap_plotter.add_spectrum(u, v, iteration)

# Plot the heatmap
heatmap_plotter.plot_heatmap()