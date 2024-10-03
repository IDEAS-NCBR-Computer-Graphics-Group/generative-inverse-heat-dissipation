
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from PIL import Image
import cv2


# def compute_wavenumbers(u, v, Lx, Ly):
#     Nx, Ny = u.shape
#     kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
#     ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
#     k = np.sqrt(kx**2 + ky**2)
#     return k

#     ############333
#     # #TODO: this is calculated every time:(
#     # # Grid dimensions
#     # Nx, Ny = u.shape

#     # # Corresponding wavenumbers
#     # kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
#     # ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
#     # k = np.sqrt(kx**2 + ky**2)

#     # # Sort the spectrum by wavenumber
#     # idx = np.argsort(k)
#     # k_sorted = k[idx]
#     # #######################  

# def compute_kolmogorov_spectrum(u, v, idx):
#     """
#     Compute and plot the Kolmogorov spectrum using the straightforward energy spectrum calculation.

#     Args:
#         u: 2D array of x-velocity fluctuations.
#         v: 2D array of y-velocity fluctuations.
#         idx: array of indexes of sorted wavenumber  sort 

#     Returns:
#         energy_spectrum: 1D array of energy spectrum values.
#     """
#     u_hat = np.fft.fft2(u)
#     v_hat = np.fft.fft2(v)
#     energy_spectrum = np.abs(u_hat)**2 + np.abs(v_hat)**2
#     energy_spectrum = np.mean(energy_spectrum, axis=0)
#     energy_spectrum_sorted = energy_spectrum[idx]
#     return  energy_spectrum_sorted

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


class KolmogorovSpectrumPlotter:
    def __init__(self, title='Energy spectrum', domain_size=(1., 1.),):
        """
        Initialize the functor with storage for the first (reference) spectrum,
        domain size, and plot title.

        Args:
            domain_size (tuple): Domain size as (Lx, Ly).
            title (str): Title for the plot.
        """
        self.first_k = None
        self.first_energy_spectrum = None
        self.domain_size = domain_size
        self.title = title

    def __call__(self, u, v):
        """
        Compute and plot the Kolmogorov spectrum for the given velocity fields.

        Args:
            u (np.ndarray): Velocity field in the x-direction.
            v (np.ndarray): Velocity field in the y-direction.
        
        Returns:
            np.ndarray: Rendered image of the spectrum plot in RGBA format.
        """
        matplotlib.use('Agg')  # Use Agg backend for rendering without GUI
        Lx, Ly = self.domain_size

        # Compute the energy spectrum for the current data
        k, energy_spectrum = compute_kolmogorov_spectrum(u, v, Lx, Ly)
        
        # If this is the first call, store the computed spectrum as the reference
        if self.first_k is None or self.first_energy_spectrum is None:
            self.first_k = k
            self.first_energy_spectrum = energy_spectrum

        # Reference data for the -5/3 slope line from the first computed spectrum
        k_ref = self.first_k[1:len(self.first_k) // 6]
        E_ref = k_ref ** (-5.0 / 3.0)
        E_ref *= self.first_energy_spectrum[10] / E_ref[0]  # Adjust scaling to match the spectrum from the first call

        # Display the histogram
        my_dpi = 100
        w, h = u.shape
        fig = plt.figure(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)
        
        # Create Axes with space for the title and labels
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size
        
        # Plot the current energy spectrum
        ax.set_ylim([1E-5, 1E2])
        ax.loglog(k[1:len(k) // 2], energy_spectrum[1:len(k) // 2], 'b>', label='Energy spectrum')
        
        # Plot the -5/3 slope line using the reference data from the first call
        ax.loglog(k_ref, E_ref, 'r--', label='-5/3 slope')

        # Plot the first computed spectrum as a reference
        ax.loglog(self.first_k[1:len(self.first_k) // 2], self.first_energy_spectrum[1:len(self.first_k) // 2], 'g.', alpha=0.5, label='Initial Energy Spectrum (Reference)')

        # Add grid and labels
        ax.grid(True, which="both", ls="--")
        ax.set_title(self.title, fontsize=12)
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_ylabel(r"Energy Spectrum $E(k)$")
        
        # Render the figure to a NumPy array
        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)  # Close the Matplotlib figure
        
        # Convert RGB to RGBA by adding an alpha channel
        canvas_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1))
        
        return canvas_rgba


# class KolmogorovSpectrumPlotter:
#     def __init__(self, title='Energy spectrum', domain_size=(1., 1.),):
#         """
#         Initialize the functor with storage for the first (reference) spectrum,
#         domain size, and plot title.

#         Args:
#             domain_size (tuple): Domain size as (Lx, Ly).
#             title (str): Title for the plot.
#         """
#         self.first_k = None
#         self.first_energy_spectrum = None
#         self.domain_size = domain_size
#         self.title = title

#     def __call__(self, u, v):
#         """
#         Compute and plot the Kolmogorov spectrum for the given velocity fields.

#         Args:
#             u (np.ndarray): Velocity field in the x-direction.
#             v (np.ndarray): Velocity field in the y-direction.
        
#         Returns:
#             np.ndarray: Rendered image of the spectrum plot in RGBA format.
#         """
#         matplotlib.use('Agg')  # Use Agg backend for rendering without GUI
#         Lx, Ly = self.domain_size

#         k = compute_wavenumbers(u, v, Lx, Ly)
#         idx = np.argsort(k)
#         k_sorted = k[idx]
#         energy_spectrum = compute_kolmogorov_spectrum(u, v, idx)
        
#         # If this is the first call, store the computed spectrum as the reference
#         if self.first_k is None or self.first_energy_spectrum is None:
#             self.first_k = k_sorted
#             self.first_energy_spectrum = energy_spectrum

#         # Reference data for the -5/3 slope line from the first computed spectrum
#         k_ref = self.first_k[1:len(self.first_k) // 6]
#         E_ref = k_ref ** (-5.0 / 3.0)
#         E_ref *= self.first_energy_spectrum[10] / E_ref[0]  # Adjust scaling to match the spectrum from the first call

#         # Display the histogram
#         my_dpi = 100
#         w, h = u.shape
#         fig = plt.figure(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)
        
#         # Create Axes with space for the title and labels
#         ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size
        
#         # Plot the current energy spectrum
#         ax.set_ylim([1E-5, 1E2])
#         ax.loglog(k_sorted[1:len(k_sorted) // 2], energy_spectrum[1:len(k) // 2], 'b>', label='Energy spectrum')
        
#         # Plot the -5/3 slope line using the reference data from the first call``
#         ax.loglog(k_ref, E_ref, 'r--', label='-5/3 slope')

#         # Plot the first computed spectrum as a reference
#         ax.loglog(self.first_k[1:len(self.first_k) // 2], self.first_energy_spectrum[1:len(self.first_k) // 2], 'g.', alpha=0.5, label='Initial Energy Spectrum (Reference)')

#         # Add grid and labels
#         ax.grid(True, which="both", ls="--")
#         ax.set_title(self.title, fontsize=12)
#         ax.set_xlabel(r"Wavenumber $k$")
#         ax.set_ylabel(r"Energy Spectrum $E(k)$")
        
#         # Render the figure to a NumPy array
#         fig.canvas.draw()
#         canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
#         plt.close(fig)  # Close the Matplotlib figure
        
#         # Convert RGB to RGBA by adding an alpha channel
#         canvas_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1))
        
#         return canvas_rgba
    
 
class SpectrumHeatmapPlotter:
    k_values = None  # Class variable shared by all instances

    def __init__(self, buffer_size, domain_size=(1.0, 1.0), target_shape=(256, 256), title = " "):
        self.domain_size = domain_size
        self.buffer_size = buffer_size
        self.target_shape = target_shape
        self.spectrums = []
        self.iterations = []
        self.title = title

    def add_spectrum(self, u, v, iteration):
        Lx, Ly = self.domain_size
        k, energy_spectrum = compute_kolmogorov_spectrum(u, v, Lx, Ly)

        if SpectrumHeatmapPlotter.k_values is None:
            SpectrumHeatmapPlotter.k_values = k

        energy_spectrum = np.interp(SpectrumHeatmapPlotter.k_values, k, energy_spectrum)
        self.spectrums.append(energy_spectrum)
        self.iterations.append(iteration)

        if len(self.spectrums) > self.buffer_size:
            self.spectrums.pop(0)
            self.iterations.pop(0)

    def plot_heatmap_rgba(self):
        """
        Plot a heatmap of the energy spectrum over iterations and return it as an RGBA image array.
        The image is generated directly at the target resolution to maintain quality.
        """


        spectrums_array = np.array(self.spectrums).T
        m = spectrums_array.shape[0]


        # Create the figure with the exact target size and DPI
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)

        # to small floats are not rendered nicely
        eps = 1E-9
        spectrums_array = np.where(spectrums_array < eps, eps, spectrums_array) 
        
        # Plot the heatmap
        cax = ax.imshow(
            spectrums_array,
            aspect='auto',
            origin='lower',
            norm=LogNorm(),  # Log scale for better visualization
            cmap='inferno'
        )

        # Add a colorbar with a label
        plt.colorbar(cax)

        # Set titles and labels
        ax.set_title(f"Spectrum {self.title} heatmap")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Wavenumber $k$")

        plt.tight_layout(pad=0.5)

        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize the canvas to 256x256 pixels if necessary
        if canvas.shape[0] != m or canvas.shape[1] != m:
            canvas = cv2.resize(canvas, (m,m), interpolation=cv2.INTER_AREA)
        
        plt.close(fig)  # Close the Matplotlib figure
        rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1)) 

        return rgba
