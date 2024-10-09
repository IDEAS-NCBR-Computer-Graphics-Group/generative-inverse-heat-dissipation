import ml_collections
import numpy as np
import torch

def get_config():
    config = ml_collections.ConfigDict()
    config.data = data = ml_collections.ConfigDict()
    data.showcase_comparison = True
    data.process_pairs = True
    data.min_init_gray_scale = 0.95
    data.max_init_gray_scale = 1.05
    data.processed_filename = 'lbm_ns_turb_pairs' if config.data.process_pairs else 'lbm_ns_turb'
    data.dataset = 'CORRUPTED_NS_MNIST'


    config.solver = solver = ml_collections.ConfigDict()
    solver.type = 'fluid'
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.domain_size = (1.0, 1.0)
    solver.turb_intensity = 1E-4
    solver.noise_limiter = (-1E-3, 1E-3)
    solver.dt_turb = 5 * 1E-4
    solver.k_min = 2.0 * torch.pi / min(solver.domain_size)
    solver.k_max = 2.0 * torch.pi / (min(solver.domain_size) / 1024)
    solver.energy_spectrum = 3 #TODO #lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    solver.min_steps = 1
    solver.max_steps = 10
    solver.n_denoising_steps = 10

    return config