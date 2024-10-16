import numpy as np
import os
import torch
from abc import ABC
import warnings

from numerical_solvers.solvers.img_reader import normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver    
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.LBM_Base_Corruptor import LBM_Base_Corruptor

class LBM_NS_Corruptor(LBM_Base_Corruptor):
    def __init__(self, config, transform=None, target_transform=None):
        super(LBM_NS_Corruptor, self).__init__(config, transform, target_transform)
        
        grid_size = (config.data.image_size, config.data.image_size)
        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            config.turbulence.domain_size, grid_size, 
            config.turbulence.turb_intensity, config.turbulence.noise_limiter,
            energy_spectrum=config.turbulence.energy_spectrum, 
            frequency_range={'k_min': config.turbulence.k_min, 'k_max': config.turbulence.k_max}, 
            dt_turb=config.turbulence.dt_turb, 
            is_div_free=config.turbulence.is_divergence_free
        )
        
        # LBM NS Solver setup        
        # Instantiate the LBM NS Solver using the config and spectral turbulence generator
        self.solver = LBM_NS_Solver(
            grid_size,
            config.solver.niu, config.solver.bulk_visc,
            spectralTurbulenceGenerator
        )
