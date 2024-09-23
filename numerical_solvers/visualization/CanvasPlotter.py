import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import cv2

from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase
from numerical_solvers.visualization.KolmogorovSpectrumPlotter import KolmogorovSpectrumPlotter


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
    
    def make_frame(self):
        
        # first row - rho
        rho_cpu = self.solver.rho.to_numpy()
        rho_img = self.render_rho(rho_cpu)
        if self.is_rho_checked:
            rho_energy_spectrum = self.render_rho_energy_spectrum(rho_cpu) 
        else:
            rho_energy_spectrum = self.dummy_canvas

        # second row - rho
        vel_cpu = self.solver.vel.to_numpy()
        vel_img = self.render_vel_mag(vel_cpu)

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
        
        # rho_histogram_rgb = make_canvas_histogram(rho_cpu, gray_min, gray_max)
        # rho_histogram_rgba = cm.ScalarMappable().to_rgba(np.flip(np.transpose(rho_histogram_rgb, (1, 0, 2)), axis=1)) 
    
        img_col1 = np.concatenate((rho_img, vel_img, force_img), axis=1)
        img_col2 = np.concatenate((rho_energy_spectrum, vel_energy_spectrum, force_energy_spectrum), axis=1)
        
        img = np.concatenate((img_col1, img_col2), axis=0)
        return img

    def write_canvas_to_file(self, img, filepath):
        # Convert image to uint8 format
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2BGR)
        print(f"cv2 writing to file {filepath}")
        cv2.imwrite(filepath, np.rot90(img_bgr))
    
    
def make_canvas_histogram(image_array, min_val, max_val):
    matplotlib.use('Agg')
    # Calculate the histogram
    
    uint8_image = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    counts, bins = np.histogram(uint8_image, bins=256, range=(0, 255))

    # Calculate the probability distribution
    total_pixels = uint8_image.size
    probability_distribution = counts / total_pixels

    # Calculate the entropy
    entropy = -np.sum([p * np.log2(p) for p in probability_distribution if p > 0])

    # Display the histogram
    my_dpi = 100
    w, h = uint8_image.shape
    fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    
    # Create Axes with space for the title and labels
    # ax = fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] as fractions of the figure size
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height] as fractions of the figure size
    
    # Plot the histogram
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 0.015])
    ax.hist(uint8_image.flatten(), bins=256, density=True)

    # Set title and labels
    # ax.text(10, 10, "Sample Text", color='white', fontsize=14, ha='left', va='top')

    ax.set_title(f'Entropy = {entropy:.4f} bits', fontsize=9) 
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    
    # plt.show() 
    # Render the figure to a NumPy array
    fig.canvas.draw()
    canvas = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # Close the Matplotlib figure
    
    # Convert RGB to RGBA by adding an alpha channel
    # canvas_rgba = np.concatenate([canvas, 255 * np.ones((*canvas.shape[:2], 1), dtype=np.uint8)], axis=2)

    # cv2.imwrite(f'output/histogram.jpg', canvas_rgba)
    return canvas