import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class MSEPlotter:
    def __init__(self, title='MSE vs. Iterations'):
        """
        Initialize the plotter with a title for the plot.
        
        Args:
            title (str): Title for the plot.
        """
        self.title = title

    def __call__(self, iterations, mse_values):
        """
        Plot the Mean Squared Error (MSE) with respect to the number of iterations.

        Args:
            iterations (np.ndarray): Array of iteration numbers.
            mse_values (np.ndarray): Array of MSE values corresponding to each iteration.
        
        Returns:
            np.ndarray: Rendered image of the MSE plot in RGBA format.
        """
        # Check if the input arrays are valid
        if len(iterations) != len(mse_values):
            raise ValueError("The length of iterations and MSE values must be the same.")

        # Initialize the plot with desired resolution
        my_dpi = 50
        fig = plt.figure(figsize=(256 / my_dpi, 256 / my_dpi), dpi=my_dpi)  # 256x256 pixels
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size

        # Plot MSE vs. Iterations
        ax.plot(iterations, mse_values, 'b-', marker='o', label='MSE')
        
        # Set logarithmic scale for better visualization, if needed
        ax.set_yscale('log')  
        
        # Add grid, title, and labels
        ax.grid(True, which="both", ls="--")
        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.legend(loc='best')

        # Render the figure to a NumPy array
        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)  # Close the Matplotlib figure
        
        # Convert RGB to RGBA by adding an alpha channel
        canvas_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1))
        
        return canvas_rgba



class SSIMPlotter:
    def __init__(self, title='SSIM vs. Iterations'):
        """
        Initialize the plotter with a title for the plot.
        
        Args:
            title (str): Title for the plot.
        """
        self.title = title

    def __call__(self, iterations, ssim_values):
        """
        Plot the Structural Similarity Index (SSIM) with respect to the number of iterations.

        Args:
            iterations (np.ndarray): Array of iteration numbers.
            ssim_values (np.ndarray): Array of SSIM values corresponding to each iteration.
        
        Returns:
            np.ndarray: Rendered image of the SSIM plot in RGBA format.
        """
        # Check if the input arrays are valid
        if len(iterations) != len(ssim_values):
            raise ValueError("The length of iterations and SSIM values must be the same.")

        # Initialize the plot with desired resolution
        my_dpi = 50
        fig = plt.figure(figsize=(256 / my_dpi, 256 / my_dpi), dpi=my_dpi)  # 256x256 pixels
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size

        # Plot SSIM vs. Iterations
        ax.plot(iterations, ssim_values, 'g-', marker='o', label='SSIM')
        
        # Set y-axis limits for better visualization
        ax.set_ylim([0.96, 1])  # SSIM ranges from 0 to 1
        
        # Add grid, title, and labels
        ax.grid(True, which="both", ls="--")
        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Structural Similarity Index (SSIM)')
        ax.legend(loc='best')

        # Render the figure to a NumPy array
        fig.canvas.draw()
        canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)  # Close the Matplotlib figure
        
        # Convert RGB to RGBA by adding an alpha channel
        canvas_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(canvas, (1, 0, 2)), axis=1))
        
        return canvas_rgba
