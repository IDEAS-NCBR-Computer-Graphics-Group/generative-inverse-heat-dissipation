
import taichi as ti
import numpy as np
from pathlib import Path

from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver
from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter

    
def run_with_gui(solver: LBM_NS_Solver, np_init_gray_image, iter_per_frame, show_gui=True):
    window = ti.ui.Window('CG - Renderer', res=(6*solver.nx, 3 * solver.ny))
    gui = window.get_gui()
    canvas = window.get_canvas()
    
    canvasPlotter = CanvasPlotter(solver, (1.0*np_init_gray_image.min(), 1.0*np_init_gray_image.max()))
    
    # warm up
    solver.solve(iterations=10)   
    solver.iterations_counter=0 # reset counter
    img = canvasPlotter.make_frame()
    
    Path("output/").mkdir(parents=True, exist_ok=True)
    canvasPlotter.write_canvas_to_file(img, f'output/iteration_{solver.iterations_counter}.jpg')
       
    while window.running:
        with gui.sub_window('MAIN MENU', x=0, y=0, width=0.2, height=0.15):
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
            
                              
        solver.solve(iter_per_frame)      
        img = canvasPlotter.make_frame()
    
        canvas.set_image(img.astype(np.float32))
        
        window.show()
        
        # if solver.iterations_counter % 500 ==0:
        #     window.save_image(f'output/iteration_{solver.iterations_counter}.jpg')
        