
import taichi as ti
import numpy as np
from pathlib import Path

from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver
from numerical_solvers.visualization.CanvasPlotter import CanvasPlotter

    
def run_with_gui(solver: LBM_NS_Solver, np_init_gray_image, iter_per_frame, show_gui=True):
    window = ti.ui.Window('CG - Renderer', res=(2*solver.nx, 3 * solver.ny))
    gui = window.get_gui()
    canvas = window.get_canvas()
    
    canvasPlotter = CanvasPlotter(solver, (1.0*np_init_gray_image.min(), 1.0*np_init_gray_image.max()))
    
    # warm up
    solver.solve(iterations=5)   
    solver.iterations_counter=0 # reset counter
    img = canvasPlotter.make_frame()
    
    Path("output/").mkdir(parents=True, exist_ok=True)
    canvasPlotter.write_canvas_to_file(img, f'output/iteration_{solver.iterations_counter}.jpg')
       
    while window.running:
        with gui.sub_window('MAIN MENU', x=0, y=0, width=0.2, height=0.1):
            # if gui.button('option1'): print("ach")
            # if gui.button('option2'): print("och")
            canvasPlotter.is_f_checked = gui.checkbox('plot f', canvasPlotter.is_f_checked)
            canvasPlotter.is_u_checked = gui.checkbox('plot u', canvasPlotter.is_u_checked)
            canvasPlotter.is_rho_checked = gui.checkbox('plot rho', canvasPlotter.is_rho_checked)
                              
        solver.solve(iter_per_frame)      
        img = canvasPlotter.make_frame()
    
        canvas.set_image(img.astype(np.float32))
        window.show()
        
        # window.save_image(f'output/iteration_{solver.iterations_counter}.jpg')
        