import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import maxwell
import matplotlib

import matplotlib.cm as cm


class VelocityHistogramPlotter:
    def __init__(self, title='Velocity Distribution (Boltzmann)'):
        """
        Initialize the plotter with a title for the plot.
        
        Args:
            title (str): Title for the plot.
        """
        self.title = title

    def __call__(self, velocity_vectors, bins=50, temperature=1.0, mass=1.0):
        """
        Plot the histogram of velocity magnitudes and the theoretical Boltzmann distribution.

        Args:
            velocity_vectors (np.ndarray): Array of velocity vectors, shape (n_particles, n_dimensions).
            bins (int): Number of bins for the histogram.
            temperature (float): Temperature parameter for the Boltzmann distribution.
            mass (float): Mass of the particles for the Boltzmann distribution.
        
        Returns:
            np.ndarray: Rendered image of the velocity histogram plot in RGBA format.
        """
        # Calculate the magnitudes of the velocity vectors
        if velocity_vectors.ndim != 2:
            raise ValueError("Input 'velocity_vectors' must be a 2D array with shape (n_particles, n_dimensions).")
        
        speeds = np.linalg.norm(velocity_vectors, axis=1)  # Ensure speeds is a 1D array
        # print(speeds.shape)

        # Calculate the theoretical Boltzmann distribution
        v = np.linspace(0, np.max(speeds), 500)
        boltzmann_dist = maxwell.pdf(v, scale=np.sqrt(temperature / mass))

        # Initialize the plot with desired resolution
        my_dpi = 50
        fig = plt.figure(figsize=(256 / my_dpi, 256 / my_dpi), dpi=my_dpi)  # 256x256 pixels
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size

        # Plot the histogram of speeds
        ax.hist(speeds, bins=bins, density=True, alpha=0.6, color='b', label='Velocity Magnitude Histogram')

        # Plot the theoretical Boltzmann distribution
        ax.plot(v, boltzmann_dist, 'r-', lw=2, label='Boltzmann Distribution')

        # Add grid, title, and labels
        ax.grid(True, which="both", ls="--")
        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel('Speed')
        ax.set_ylabel('Probability Density')
        ax.legend(loc='best')

        # Render the figure to a NumPy array
        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)   # Close the Matplotlib figure
        
        # Convert RGB to RGBA by adding an alpha channel
        canvas_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1))
        
        return canvas_rgba