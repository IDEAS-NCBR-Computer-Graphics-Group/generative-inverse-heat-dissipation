import ml_collections
from configs.ffhq import default_ffhq_configs
import numpy as np
import torch

def get_config():
    return get_default_configs()

def get_default_configs():
    config = default_ffhq_configs.get_default_configs()
    model = config.model
    config.data.image_size = 128
    config.data.dataset = 'FFHQ_128' 
    config.training.batch_size = 4

    model.blur_sigma_max = 20
    model.blur_sigma_min = 0.5
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
    model.K = 50
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    
    training = config.training
    training.log_freq = 50
    training.eval_freq = 100
    training.n_iters = 20001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.sampling_freq = 100

    # lbm
    config.lbm = lbm = ml_collections.ConfigDict()

    lbm.data = data = ml_collections.ConfigDict()
    data.min_init_gray_scale = 0.95
    data.max_init_gray_scale = 1.05
    config.data.processed_filename = 'lbm_ns_turb_pairs' if config.data.process_pairs else 'lbm_ns_turb'
    
    config.lbm.solver = solver = ml_collections.ConfigDict()
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.domain_size = (1.0, 1.0)
    solver.turb_intensity = 1E-4
    solver.noise_limiter = (-1E-3, 1E-3)
    solver.dt_turb = 5 * 1E-4
    solver.min_lbm_steps = 1
    solver.max_lbm_steps = 10
    solver.k_min = 2.0 * torch.pi / min(solver.domain_size)
    solver.k_max = 2.0 * torch.pi / (min(solver.domain_size) / 1024)
    solver.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    solver.n_denoising_steps = 10

    # blur
    config.blur = blur = ml_collections.ConfigDict()
    blur.data = data = ml_collections.ConfigDict()

    data.min_init_gray_scale = 0.0
    data.max_init_gray_scale = 1.0
    data.processed_filename = 'blurr_pairs' if config.data.process_pairs else 'blurr'


    blur.solver = solver = ml_collections.ConfigDict()
    solver.min_blurr = 1.
    solver.max_blurr = 5.
    solver.n_denoising_steps = 10

    
    return config
