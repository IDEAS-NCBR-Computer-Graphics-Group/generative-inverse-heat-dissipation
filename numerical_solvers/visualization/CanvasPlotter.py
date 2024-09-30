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
from scipy.stats import norm, chi

from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase
from numerical_solvers.visualization.KolmogorovSpectrumPlotter import KolmogorovSpectrumPlotter, SpectrumHeatmapPlotter
from numerical_solvers.visualization.MetricPlotter import MSEPlotter, SSIMPlotter


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
        self.is_v_distribution_checked = False
        self.is_heatmap_checked = False
        self.is_rho_diff_checked = False
        self.is_energy_diff_checked = False

        self.is_vel_mag_distribution_checked = False

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

        # if self.solver.iterations_counter % 10 ==0:
        #     print("rho max:  ", rho_cpu.max())
        #     print("rho min:  ", rho_cpu.min())



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
        vx = vel_cpu[:,:,0]
        vy = vel_cpu[:,:,1]
        vel_img = self.render_vel_mag(vel_cpu)
        vel_mag_img = self.return_vel_mag(vel_cpu)

        
        
        image1 = img_as_float(self.energy_history.buffer[self.energy_history.index - 2])
        image2 = img_as_float(vel_mag_img)
        ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())
        self.energy_history.add_snapshot(vel_mag_img, mean_squared_error(self.energy_history.buffer[self.energy_history.index - 2], vel_mag_img), ssim_index, self.solver.iterations_counter) 

        if self.is_u_checked:
            vel_energy_spectrum = self.render_vel_energy_spectrum(vel_cpu)
        else:
            # vel_energy_spectrum = self.compute_divergence(vel_cpu[:, :, 0], vel_cpu[:, :, 1])
            vel_energy_spectrum = self.dummy_canvas
    
        force_cpu = self.solver.Force.to_numpy()
        force_img = self.render_force_mag(force_cpu)
        if self.is_f_checked:
            force_energy_spectrum = self.render_force_energy_spectrum(force_cpu)
        else:
            force_energy_spectrum =  self.dummy_canvas   
        
        

        # third row - heatmap + differences

        self.heatmap_plotter.add_spectrum(vel_cpu[:, :, 0], vel_cpu[:, :, 1], self.solver.iterations_counter)
        if self.is_heatmap_checked:
            
            heatmap_energy  = self.heatmap_plotter.plot_heatmap_rgba()
        else:
            heatmap_energy = self.dummy_canvas

        img_energy_difference = None
        if self.solver.iterations_counter < 2:
            if img_energy_difference is None:  
                img_energy_difference = self.dummy_canvas
                img_rho_difference = self.dummy_canvas
        else:
            
                # Update img_energy_difference only on every second step
                if self.is_rho_diff_checked:
                    rho_difference = self.rho_history.buffer[self.rho_history.index - 2] - rho_cpu
                    img_rho_difference  = self.render_energy_difference(rho_difference)
                else:
                    img_rho_difference = self.dummy_canvas

                if self.is_energy_diff_checked:
                    energy_difference = self.energy_history.buffer[self.energy_history.index - 2] - vel_mag_img        
                    img_energy_difference = self.render_energy_difference(energy_difference)
                else:
                    img_energy_difference = self.dummy_canvas

                image1 = img_as_float(self.energy_history.buffer[self.energy_history.index - 2])
                image2 = img_as_float(vel_mag_img)
                ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())

                image1 = img_as_float(self.rho_history.buffer[self.rho_history.index - 2])
                image2 = img_as_float(rho_cpu)
                ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())

        if self.is_vel_mag_distribution_checked:
            # border = 2
            # vel_mag_img[border:-border,border:-border] =0
            # plot_velocity_magnitude_distribution
            vel_mag_histogram_rgb = plot_velocity_magnitude_distribution(vel_mag_img)
            vel_mag_histogram_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(vel_mag_histogram_rgb, (1, 0, 2)), axis=1)) 

        else:
            vel_mag_histogram_rgba = self.dummy_canvas
        
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

        
        if self.is_v_distribution_checked:
            #TODO: JJM get rid of this ugly hack
            # border = 2
            # vy = vy[border:-border,border:-border]
            # vx = vx[border:-border,border:-border]
            v_distribution_y = plot_v_component_distribution(vy, "v_y component distribution")
            v_distribution_rgba_y = cm.ScalarMappable().to_rgba(np.flip(np.transpose(v_distribution_y, (1, 0, 2)), axis=1)) 
            v_distribution_x = plot_v_component_distribution(vx, "v_x component distribution")
            v_distribution_rgba_x = cm.ScalarMappable().to_rgba(np.flip(np.transpose(v_distribution_x, (1, 0, 2)), axis=1)) 
        else:
            v_distribution_rgba_y = self.dummy_canvas
            v_distribution_rgba_x = self.dummy_canvas


    

        img_col1 = np.concatenate((rho_img, vel_img, force_img), axis=1)
        img_col2 = np.concatenate((rho_energy_spectrum, vel_energy_spectrum, force_energy_spectrum), axis=1)
        img_col3 = np.concatenate((img_rho_difference, img_energy_difference,  heatmap_energy), axis=1)
        img_col4 = np.concatenate((mse_rho_image, mse_energy_image, vel_mag_histogram_rgba), axis=1)
        img_col5 = np.concatenate((ssim_rho_image, ssim_energy_image, v_distribution_rgba_y), axis=1)
        img_col6 = np.concatenate((self.dummy_canvas, self.dummy_canvas, v_distribution_rgba_x), axis=1)


        img = np.concatenate((img_col1, img_col2, img_col3, img_col4, img_col5, img_col6), axis=0)
        return img

    def write_canvas_to_file(self, img, filepath):
        # Convert image to uint8 format
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        print(f"cv2 writing to file {filepath}")
        cv2.imwrite(filepath, np.rot90(img_bgr))
    
    


def plot_v_component_distribution(v_data, title):
    matplotlib.use('Agg')  # Use the 'Agg' backend for headless environments (no GUI)
    
    m= v_data.shape[0]
    v_data = v_data.reshape(-1)  # Flatten the data

    num_bins = 512
    hist_data, bins = np.histogram(v_data, bins=num_bins, density=True)
    # bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit Gaussian distribution to the data
    mu, sigma = norm.fit(v_data)
    
    # Create an array over the full range specified
    x_range = np.linspace(-max(v_data.max(), -v_data.min())*2, max(v_data.max(), -v_data.min())*2, 256)
    gaussian_fit = norm.pdf(x_range, mu, sigma)

    # Display the histogram and the fit
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # This will create a 256x256 pixel figure

    # Plot the histogram and Gaussian fit
    ax.hist(v_data, bins=num_bins, density=True, alpha=0.6, color='g')
    # ax.set_xlim([-max(v_data.max(), -v_data.min()), max(v_data.max(), -v_data.min())])
    ax.set_xlim(-5E-2, 5E-2)
    ax.plot(x_range, gaussian_fit, 'r-', lw=2)
    ax.set_title(f'{title}', fontsize=10)
    ax.set_xlabel('v component')
    ax.set_ylabel('Probability Density')

    plt.tight_layout(pad=1.2)
    
    # Render the figure to a NumPy array
    fig.canvas.draw()
    canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Resize the canvas to 256x256 pixels if necessary
    if canvas.shape[0] != m or canvas.shape[1] != m:
        canvas = cv2.resize(canvas, (m, m), interpolation=cv2.INTER_AREA)
    
    plt.close(fig)  # Close the Matplotlib figure

    return canvas


def plot_velocity_magnitude_distribution(vel_mag_img):
    matplotlib.use('Agg')  # Use the 'Agg' backend for headless environments (no GUI)
    
    m= vel_mag_img.shape[0]
    vel_mag_img = vel_mag_img.reshape(-1)
     
    num_bins = 64
    hist_data, bins = np.histogram(vel_mag_img, bins=num_bins, density=True)

    params = maxwell.fit(vel_mag_img)
    x_range = np.linspace(0, 0.05, 300)
    mb_fit = maxwell.pdf(x_range, *params)

    # Create the plot
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # This will create a 256x256 pixel figure

    # Plot the histogram and Maxwell-Boltzmann fit
    ax.hist(vel_mag_img, bins=bins, density=True, alpha=0.6, color='g')
    ax.plot(x_range, mb_fit, 'r-', lw=2)
    ax.set_ylim(0, 256)
    ax.set_title('Velocity Magnitude Distribution', fontsize=10)
    ax.set_xlabel('Velocity Magnitude')
    ax.set_ylabel('Probability Density')

    plt.tight_layout(pad=1.2)

    # Render the figure to a NumPy array
    fig.canvas.draw()
    canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Ensure the output is exactly 256x256 pixels
    if canvas.shape[0] != m or canvas.shape[1] != m:
        canvas = cv2.resize(canvas, (m, m), interpolation=cv2.INTER_AREA)

    plt.close(fig)  # Close the Matplotlib figure to free up resources
    
    return canvas