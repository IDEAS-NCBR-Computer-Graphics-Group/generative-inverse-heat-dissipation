import taichi as ti
import numpy as np
import torch
from tqdm import tqdm
import warnings
import os

from solvers.img_reader import normalize_grayscale_image_range
from solvers.LBM_NS_Solver import LBM_NS_Solver    
from solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from corruptors.BaseCorruptor import BaseCorruptor

class LBM_NS_Corruptor(BaseCorruptor):
    def __init__(
            self,
            config,
            transform=None,
            target_transform=None
            ):
        super(LBM_NS_Corruptor, self).__init__(transform, target_transform)

        ti.init(arch=ti.gpu)

        grid_size = (config.data.image_size, config.data.image_size)
        config = config.lbm
        # energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
        # frequency_range = {'k_min': config.solver.k_min, 'k_max': config.solver.k_max }

        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            config.solver.domain_size,
            grid_size, 
            config.solver.turb_intensity,
            config.solver.noise_limiter,
            energy_spectrum=config.solver.energy_spectrum, 
            frequency_range={'k_min': config.solver.k_min, 'k_max': config.solver.k_max}, 
            dt_turb=config.solver.dt_turb, 
            is_div_free=False
        )
        
        # LBM NS Solver setup        
        # Instantiate the LBM NS Solver using the config and spectral turbulence generator
        self.solver = LBM_NS_Solver(
            "miau",
            grid_size,
            config.solver.niu, config.solver.bulk_visc,
            spectralTurbulenceGenerator
        )

        # Set LBM steps (can be made configurable too)
        self.max_steps = config.solver.max_lbm_steps
        self.min_steps = config.solver.min_lbm_steps
        
        self.min_init_gray_scale = config.data.min_init_gray_scale
        self.max_init_gray_scale = config.data.max_init_gray_scale
        
        
    def _corrupt(self, x, lbm_steps, generate_pair=False):
        """
        Corrupts the input image using LBM solver.

        Args:
            x (torch.Tensor): The input image tensor.
            lbm_steps (int): The number of steps for LBM solver to run.
            generate_pair (bool): Flag to generate a pair of images (before and after a step difference). 
                                  Default is False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Corrupted image or a pair of corrupted images.
        """
        step_difference = 1  # Step difference for generating pairs
        np_gray_img = x.numpy()[0, :, :]
        np_gray_img = normalize_grayscale_image_range(
            np_gray_img, self.min_init_gray_scale, self.max_init_gray_scale)
        
        self.solver.init(np_gray_img)
        self.solver.iterations_counter = 0  # Reset counter

        if generate_pair:
            # Solve up to (lbm_steps - step_difference) if generating pairs
            self.solver.solve(lbm_steps - step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            less_noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            
            # Solve the remaining steps for the pair generation
            self.solver.solve(step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            return noisy_x, less_noisy_x,
        else:
            # Solve up to lbm_steps - step_difference if generating pairs, else directly to lbm_steps
            self.solver.solve(lbm_steps)
            rho_cpu = self.solver.rho.to_numpy()
            noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            return noisy_x, None
