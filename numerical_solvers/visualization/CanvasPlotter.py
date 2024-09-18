import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import cv2

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import maxwell
from scipy.optimize import curve_fit

from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase
from numerical_solvers.visualization.KolmogorovSpectrumPlotter import KolmogorovSpectrumPlotter, SpectrumHeatmapPlotter
from numerical_solvers.visualization.MetricPlotter import MSEPlotter, SSIMPlotter
from numerical_solvers.visualization.histogram_plotter import VelocityHistogramPlotter
class CircularBuffer:
    def __init__(self, buffer_size, array_shape):
        """
        Initializes the circular buffer.

        Parameters:
        - buffer_size (int): Maximum number of snapshots to store.
        - array_shape (tuple): Shape of each snapshot array (e.g., (256, 256)).
        """
        self.buffer_size = buffer_size
        self.array_shape = array_shape
        self.buffer = np.zeros((buffer_size, *array_shape), dtype=np.float32)  # Store snapshots
        self.mse_values = np.zeros(buffer_size, dtype=np.float32)  # Store MSE values
        self.ssim_values = np.zeros(buffer_size, dtype=np.float32)  # Store SSIM values
        self.iteration_numbers = np.zeros(buffer_size, dtype=np.int32)  # Store iteration numbers
        self.index = 0
        self.full = False

    def add_snapshot(self, snapshot, mse, ssim, iteration):
        """
        Adds a new snapshot along with its MSE, SSIM values, and iteration number to the circular buffer.

        Parameters:
        - snapshot (ndarray): A new snapshot array with shape equal to `array_shape`.
        - mse (float): The Mean Squared Error value corresponding to this snapshot.
        - ssim (float): The Structural Similarity Index Measure value corresponding to this snapshot.
        - iteration (int): The iteration number corresponding to this snapshot.
        """
        if snapshot.shape != self.array_shape:
            raise ValueError(f"Snapshot shape must be {self.array_shape}, but got {snapshot.shape}.")
        
        self.buffer[self.index] = snapshot
        self.mse_values[self.index] = mse
        self.ssim_values[self.index] = ssim
        self.iteration_numbers[self.index] = iteration
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.full = True

    def get_snapshots(self):
        """
        Retrieves all snapshots in the buffer in the order they were added.
        
        Returns:
        - ndarray: An array of snapshots.
        """
        if self.full:
            return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]), axis=0)
        else:
            return self.buffer[:self.index]
    
    def get_metrics(self):
        """
        Retrieves MSE and SSIM values in the buffer in the order they were added.
        
        Returns:
        - Tuple (ndarray, ndarray): Arrays of MSE and SSIM values.
        """
        if self.full:
            mse_values = np.concatenate((self.mse_values[self.index:], self.mse_values[:self.index]), axis=0)
            ssim_values = np.concatenate((self.ssim_values[self.index:], self.ssim_values[:self.index]), axis=0)
        else:
            mse_values = self.mse_values[:self.index]
            ssim_values = self.ssim_values[:self.index]
        
        return mse_values, ssim_values
    
    def get_iterations(self):
        """
        Retrieves iteration numbers in the buffer in the order they were added.
        
        Returns:
        - ndarray: An array of iteration numbers.
        """
        if self.full:
            return np.concatenate((self.iteration_numbers[self.index:], self.iteration_numbers[:self.index]), axis=0)
        else:
            return self.iteration_numbers[:self.index]




class CanvasPlotter:
    def __init__(self, solver: LBM_SolverBase, gray_range):
        
        zera = np.zeros((solver.nx, solver.ny))
        self.dummy_canvas = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap="gist_gray").to_rgba(zera)
        
        self.solver = solver 
        self.gray_min, self.gray_max = gray_range
        
        self.rho_kolmogorov_plotter = KolmogorovSpectrumPlotter(title='rho spectrum')
        self.u_kolmogorov_plotter = KolmogorovSpectrumPlotter(title='u spectrum')
        self.force_kolmogorov_plotter = KolmogorovSpectrumPlotter(title='force spectrum')

        
        self.is_rho_checked = False
        self.is_u_checked = False
        self.is_f_checked = False
        self.is_rho_MSE_checked = False
        self.is_rho_SSIM_checked = False

        self.is_energy_MSE_checked = False
        self.is_energy_SSIM_checked = False

        self.buffer_size = 1000
        self.array_shape = (solver.nx, solver.ny)
        self.energy_history = CircularBuffer(self.buffer_size, self.array_shape)
        self.rho_history = CircularBuffer(self.buffer_size, self.array_shape)

        self.heatmap_plotter = SpectrumHeatmapPlotter(self.buffer_size)
    
    def compute_divergence(self, u, v, dx=1, dy=1):
        """
        Computes the divergence of a 2D vector field (u, v) using finite difference stencils.
        
        Parameters:
        - u: 2D array of x-velocity (Ny, Nx)
        - v: 2D array of y-velocity (Ny, Nx)
        - dx: Grid spacing in the x-direction
        - dy: Grid spacing in the y-direction
        
        Returns:
        - div: 2D array of divergence values (Ny, Nx)
        """        
        div_u = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
        div_v = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
        div = div_u + div_v

        div_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1E-3, vmax=1E-3),cmap="coolwarm").to_rgba(div)
        
        return div_img
    
    def return_vel_mag(self, vel):
        return np.sqrt((vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2))

    def render_vel_mag(self, vel):
        vel_mag = np.sqrt((vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2))
        vel_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.05), cmap="coolwarm").to_rgba(vel_mag)
        return vel_img
    
    def render_vel_energy_spectrum(self, vel):
        return self.u_kolmogorov_plotter(vel[:, :, 0], vel[:, :, 1])
    
    def render_force_mag(self,force):
        force_mag = np.sqrt((force[:, :, 0] ** 2 + force[:, :, 1] ** 2))
        force_mag = cm.ScalarMappable(cmap="inferno").to_rgba(force_mag)
        return force_mag

    def render_force_energy_spectrum(self, force):
        return self.force_kolmogorov_plotter(force[:, :, 0], force[:, :, 1])
    
    def render_rho(self, rho_cpu):
        rho_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=self.gray_min, vmax=self.gray_max), cmap="gist_gray").to_rgba(rho_cpu) 
        return rho_img
    
    def render_rho_energy_spectrum(self, rho_cpu):
        rho_energy_spectrum = self.rho_kolmogorov_plotter(rho_cpu, np.zeros_like(rho_cpu))
        return rho_energy_spectrum
    
    def render_energy_difference(self, energy_difference):
        energy_diff_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=0.01), cmap="gist_gray").to_rgba(energy_difference) 
        return energy_diff_img
    
    def render_rho_difference(self, rho_difference):
        rho_diff_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=0.01), cmap="gist_gray").to_rgba(rho_difference) 
        return rho_diff_img

    def make_frame(self):
        
        # first row - rho
        
        rho_cpu = self.solver.rho.to_numpy()
        rho_img = self.render_rho(rho_cpu)
        image1 = img_as_float(self.rho_history.buffer[self.rho_history.index - 2])
        image2 = img_as_float(rho_cpu)
        ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())
        self.rho_history.add_snapshot(rho_cpu, mean_squared_error(self.rho_history.buffer[self.rho_history.index - 2], rho_cpu), ssim_index, self.solver.iterations_counter) 

        if self.is_rho_checked:
            rho_energy_spectrum = self.render_rho_energy_spectrum(rho_cpu) 
        else:
            rho_energy_spectrum = self.dummy_canvas

        # second row - rho

        vel_cpu = self.solver.vel.to_numpy()
        vel_img = self.render_vel_mag(vel_cpu)
        vel_mag_img = self.return_vel_mag(vel_cpu)
        image1 = img_as_float(self.energy_history.buffer[self.energy_history.index - 2])
        image2 = img_as_float(vel_mag_img)
        ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())
        self.energy_history.add_snapshot(vel_mag_img, mean_squared_error(self.energy_history.buffer[self.energy_history.index - 2], vel_mag_img), ssim_index, self.solver.iterations_counter) 

        if self.is_u_checked:
            vel_energy_spectrum = self.render_vel_energy_spectrum(vel_cpu)
        else:
            vel_energy_spectrum = self.compute_divergence(vel_cpu[:, :, 0], vel_cpu[:, :, 1])
    
        force_cpu = self.solver.Force.to_numpy()
        force_img = self.render_force_mag(force_cpu)
        if self.is_f_checked:
            force_energy_spectrum = self.render_force_energy_spectrum(force_cpu)
        else:
            force_energy_spectrum =  self.dummy_canvas   
        
        # rho_histogram_rgb = make_canvas_histogram(rho_cpu, rho_cpu.min(), rho_cpu.max(), "rho")
        # rho_histogram_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(rho_histogram_rgb, (1, 0, 2)), axis=1)) 

        self.heatmap_plotter.add_spectrum(vel_cpu[:, :, 0], vel_cpu[:, :, 1], self.solver.iterations_counter)
        heatmap_energy  = self.heatmap_plotter.plot_heatmap_rgba()
        # third row - metrics

        # Initialize img_energy_difference outside the rendering loop to hold its value between updates
        img_energy_difference = None  # Start with None or an appropriate initial image

        if self.solver.iterations_counter < 2:
            # For the first two iterations, initialize with a zero image or keep the last known value
            if img_energy_difference is None:  # Only set initially if it's not already initialized
                img_energy_difference = self.render_rho(np.zeros(vel_img.shape))
                img_rho_difference = self.render_rho(np.zeros(vel_img.shape))
                
        else:
            
                # Update img_energy_difference only on every second step
                energy_difference = self.energy_history.buffer[self.energy_history.index - 2] - vel_mag_img
                rho_difference = self.rho_history.buffer[self.rho_history.index - 2] - rho_cpu

                if self.solver.iterations_counter%10 ==0: 
                    print("MSE energy: ", mean_squared_error(self.energy_history.buffer[self.energy_history.index - 2], vel_mag_img))
                    print("MSE rho", mean_squared_error(self.rho_history.buffer[self.rho_history.index - 2], rho_cpu))

                image1 = img_as_float(self.energy_history.buffer[self.energy_history.index - 2])
                image2 = img_as_float(vel_mag_img)
                ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())

                if self.solver.iterations_counter%10 ==0: 
                    print("energy ssim", ssim_index)

                image1 = img_as_float(self.rho_history.buffer[self.rho_history.index - 2])
                image2 = img_as_float(rho_cpu)
                ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())

                if self.solver.iterations_counter%10 ==0: 
                    print("rho ssim", ssim_index)


                img_energy_difference = self.render_energy_difference(energy_difference)
                img_rho_difference  = self.render_energy_difference(rho_difference)
                # print(img_energy_difference.min())

        # Ensure img_energy_difference is not None at the point of use
        if img_energy_difference is None:
            img_energy_difference = np.zeros(vel_img.shape)


        vel_histogram_rgb = make_canvas_histogram(vel_mag_img, vel_mag_img.min(), vel_mag_img.max(), "velocity")
        vel_histogram_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(vel_histogram_rgb, (1, 0, 2)), axis=1)) 



        # n_particles = 256 * 256  # Total number of "particles" or grid points
        # vel_cpu_reshaped = vel_cpu.reshape((n_particles, 2))  # Shape becomes (65536, 2)
        # velocity_plotter = VelocityHistogramPlotter(title='Vel')
        # velocity_histogram_image = velocity_plotter(vel_cpu_reshaped, bins=50, temperature=1.0, mass=1.0)
        # print(velocity_histogram_image.shape)
        
        # fourth row

        if self.solver.iterations_counter < 2:
            # For the first two iterations, initialize with a zero image or keep the last known value
            mse_rho_image = self.dummy_canvas
            ssim_rho_image = self.dummy_canvas

            mse_energy_image = self.dummy_canvas
            ssim_energy_image = self.dummy_canvas

        else:
            if self.is_rho_MSE_checked:
                mse_values, ssim_values = self.rho_history.get_metrics()
                iteration_numbers = self.rho_history.get_iterations()
                mse_plotter = MSEPlotter(title='rho - MSE(Iterations)')
                mse_rho_image = mse_plotter(iteration_numbers, mse_values)
            else:
                mse_rho_image = self.dummy_canvas

            if self.is_rho_SSIM_checked:
                mse_values, ssim_values = self.rho_history.get_metrics()
                iteration_numbers = self.rho_history.get_iterations()
                ssim_plotter = SSIMPlotter(title='rho - SSIM(Iterations)')
                ssim_rho_image = ssim_plotter(iteration_numbers,ssim_values)
            else:
                ssim_rho_image = self.dummy_canvas

            if self.is_energy_MSE_checked:
                mse_values, ssim_values = self.energy_history.get_metrics()
                iteration_numbers = self.energy_history.get_iterations()
                mse_plotter = MSEPlotter(title='Kinetic energy - MSE(Iterations)')
                mse_energy_image = mse_plotter(iteration_numbers, mse_values)
            else:
                mse_energy_image = self.dummy_canvas

            if self.is_energy_SSIM_checked:
                mse_values, ssim_values = self.energy_history.get_metrics()
                iteration_numbers = self.energy_history.get_iterations()
                ssim_plotter = SSIMPlotter(title='Kinetic energy - SSIM(Iterations)')
                ssim_energy_image = ssim_plotter(iteration_numbers,ssim_values)
            else:
                ssim_energy_image = self.dummy_canvas

        
    

        img_col1 = np.concatenate((rho_img, vel_img, force_img), axis=1)
        img_col2 = np.concatenate((rho_energy_spectrum, vel_energy_spectrum, force_energy_spectrum), axis=1)
        img_col3 = np.concatenate((img_rho_difference, img_energy_difference,  heatmap_energy), axis=1)
        img_col4 = np.concatenate((mse_rho_image, mse_energy_image, vel_histogram_rgba), axis=1)
        img_col5 = np.concatenate((ssim_rho_image, ssim_energy_image, self.dummy_canvas), axis=1)


        img = np.concatenate((img_col1, img_col2, img_col3, img_col4, img_col5), axis=0)
        return img

    def write_canvas_to_file(self, img, filepath):
        # Convert image to uint8 format
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        print(f"cv2 writing to file {filepath}")
        cv2.imwrite(filepath, np.rot90(img_bgr))
    
    
def maxwell_boltzmann(v, T):
    """
    Maxwell-Boltzmann distribution function with adjustable temperature.
    
    Parameters:
    - v: Speed
    - T: Temperature
    
    Returns:
    - Probability density at speed v.
    """
    m = 1  # Mass
    k_B = 1  # Boltzmann constant (set to 1 for simplicity)
    return np.sqrt(2/np.pi) * (m / (k_B * T))**(3/2) * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

def make_canvas_histogram(image_array, min_val, max_val, title):
    # Use the 'Agg' backend for headless environments (no GUI)
    matplotlib.use('Agg')  # This must be inside the function to be effective here

    # Calculate the histogram
    uint8_image = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    counts, bins = np.histogram(uint8_image, bins=32, range=(0, 100))

    # Calculate the probability distribution
    total_pixels = uint8_image.size
    probability_distribution = counts / total_pixels

    # Calculate the entropy
    entropy = -np.sum([p * np.log2(p) for p in probability_distribution if p > 0])

    # Display the histogram
    my_dpi = 70
    w, h = uint8_image.shape
    fig = plt.figure(figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)
    
    # Create Axes with space for the title and labels
    ax = fig.add_axes([0.18, 0.12, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size
    
    # Plot the histogram
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 0.3])
    hist_data, bins, _ = ax.hist(uint8_image.flatten(), bins=64, range=(0, 100), density=True, alpha=0.5, label='Data Histogram')

    # Generate data points for fitting within the same range as the histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Use bin centers for fitting

    # Fit the Maxwell-Boltzmann distribution to the histogram data
    popt, _ = curve_fit(maxwell_boltzmann, bin_centers, hist_data, p0=[10])  # Initial guess for T is 10

    # Generate a smooth curve for the fitted Maxwell-Boltzmann distribution
    v = np.linspace(0.1, 100, 300)  # Avoid zero to prevent division by zero
    mb_fit = maxwell_boltzmann(v, *popt)

    # Plot the fitted Maxwell-Boltzmann distribution
    ax.plot(v, mb_fit, 'r-', lw=2, label=f'Maxwell-Boltzmann Fit (T={popt[0]:.2f})')

    # Set title and labels
    ax.set_title(f'{title} Entropy = {entropy:.4f} bits', fontsize=9) 
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Render the figure to a NumPy array
    fig.canvas.draw()
    canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # Close the Matplotlib figure

    return canvas