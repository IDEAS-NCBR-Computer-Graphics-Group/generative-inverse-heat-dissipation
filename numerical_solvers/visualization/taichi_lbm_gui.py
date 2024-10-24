
import taichi as ti
import numpy as np
import time
from pathlib import Path

from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver
from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter
# from skimage import data, img_as_float
import matplotlib
import matplotlib.cm as cm

def run_simple_gui(solver: LBM_NS_Solver, np_init_gray_image, iter_per_frame, show_gui=True):
    gui_res = (1 * solver.nx, 3 * solver.ny)
    window = ti.ui.Window('CG - Renderer', res=gui_res)
    gui = window.get_gui()
    canvas = window.get_canvas()

    while window.running:
        rho_cpu = solver.rho.to_numpy()
        rho_img = cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.9*np_init_gray_image.min(), vmax=1.1*np_init_gray_image.max()),
                cmap="gist_gray").to_rgba(rho_cpu)

        vel = solver.vel.to_numpy()
        vel_mag = np.sqrt((vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2))
        vel_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=0.05), cmap="coolwarm").to_rgba(vel_mag)

        force = solver.Force.to_numpy()
        force_mag = np.sqrt((force[:, :, 0] ** 2 + force[:, :, 1] ** 2))
        force_mag = cm.ScalarMappable(cmap="inferno").to_rgba(force_mag)

        img = np.concatenate((rho_img, vel_img, force_mag), axis=1)
        canvas.set_image(img.astype(np.float32))

        solver.solve(iter_per_frame)
        window.show()

def run_with_gui(solver: LBM_NS_Solver, np_init_gray_image, iter_per_frame, show_gui=True):
    gui_res = (6*solver.nx, 3 * solver.ny)

    window = ti.ui.Window('CG - Renderer', res=gui_res)
    gui = window.get_gui()
    canvas = window.get_canvas()
    
    canvasPlotter = CanvasPlotter(solver, (0.9*np_init_gray_image.min(), 1.1*np_init_gray_image.max()))
    
    # warm up
    solver.solve(iterations=1)   
    solver.iterations_counter=0 # reset counter
    img = canvasPlotter.make_frame()
    
    Path("local_outputs/").mkdir(parents=True, exist_ok=True)
    canvasPlotter.write_canvas_to_file(img, f'local_outputs/iteration_{solver.iterations_counter}.jpg')
       
    while window.running:
        with gui.sub_window('MAIN MENU', x=0, y=0, width=0.2, height=0.25):
            # if gui.button('option1'): print("ach")
            # if gui.button('option2'): print("och")
            canvasPlotter.is_f_checked = gui.checkbox('plot f', canvasPlotter.is_f_checked)
            canvasPlotter.is_u_checked = gui.checkbox('plot u', canvasPlotter.is_u_checked)
            canvasPlotter.is_rho_checked = gui.checkbox('plot rho', canvasPlotter.is_rho_checked)
            canvasPlotter.is_rho_MSE_checked = gui.checkbox('plot rho MSE', canvasPlotter.is_rho_MSE_checked)
            canvasPlotter.is_rho_SSIM_checked = gui.checkbox('plot rho SSIM', canvasPlotter.is_rho_SSIM_checked)
            canvasPlotter.is_energy_MSE_checked = gui.checkbox('plot energy MSE', canvasPlotter.is_energy_MSE_checked)
            canvasPlotter.is_energy_SSIM_checked = gui.checkbox('plot energy SSIM', canvasPlotter.is_energy_SSIM_checked)
            canvasPlotter.is_rho_diff_checked = gui.checkbox('rho difference', canvasPlotter.is_rho_diff_checked)
            canvasPlotter.is_energy_diff_checked = gui.checkbox('energy difference', canvasPlotter.is_energy_diff_checked)
            canvasPlotter.is_v_distribution_checked = gui.checkbox('plot 1D v distribution', canvasPlotter.is_v_distribution_checked)
            
            canvasPlotter.is_heatmap_checked = gui.checkbox('heatmap', canvasPlotter.is_heatmap_checked)
            canvasPlotter.is_vel_mag_distribution_checked = gui.checkbox('plot norm(v) ', canvasPlotter.is_vel_mag_distribution_checked)
            
        img = canvasPlotter.make_frame()
        canvas.set_image(img.astype(np.float32))

        solver.solve(iter_per_frame)
        window.show()
        
        # time.sleep(4)
        # window.running = False
    
    # window.show()
    # gui.show()
        # if solver.iterations_counter % 500 ==0:
        #     window.save_image(f'local_outputs/iteration_{solver.iterations_counter}.jpg')
        