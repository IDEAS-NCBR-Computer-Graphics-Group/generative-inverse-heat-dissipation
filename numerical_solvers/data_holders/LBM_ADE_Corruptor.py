import numpy as np
import os
import torch
from abc import ABC
import warnings

import taichi as ti
from numerical_solvers.solvers.img_reader import normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_ADE_Solver import LBM_ADE_Solver
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.LBM_Base_Corruptor import LBM_Base_Corruptor

class LBM_ADE_Corruptor(LBM_Base_Corruptor):
    def __init__(self, config, transform=None, target_transform=None):
        super(LBM_ADE_Corruptor, self).__init__(config, transform, target_transform)
        ti.init(arch=ti.gpu)
        
        grid_size = (config.data.image_size, config.data.image_size)
        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            config.solver.domain_size, grid_size, 
            config.solver.turb_intensity, config.solver.noise_limiter,
            energy_spectrum=config.solver.energy_spectrum, 
            frequency_range={'k_min': config.solver.k_min, 'k_max': config.solver.k_max}, 
            dt_turb=config.solver.dt_turb, 
            is_div_free=config.solver.is_divergence_free
        )
        
        # LBM NS Solver setup        
        # Instantiate the LBM NS Solver using the config and spectral turbulence generator
        self.solver = LBM_ADE_Solver(
            grid_size,
            config.solver.niu, config.solver.bulk_visc,
            spectralTurbulenceGenerator
        )