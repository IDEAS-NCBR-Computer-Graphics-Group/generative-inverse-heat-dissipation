from configs.mnist import default_lbm_mnist_configs as default_mnist_configs
import ml_collections
import numpy as np
import torch
from torchvision import transforms
from configs.conf_utils import hash_solver

def get_config():
    config = default_mnist_configs.get_default_configs()
    
    model = config.model
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
        
    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.process_all = True
    data.processed_filename = 'lbm_ade_turb_pairs' if config.data.process_pairs else 'lbm_ade_turb'
    data.dataset = 'CORRUPTED_NS_MNIST'
    
    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100
    data.transform = transforms.Compose([])
    
    config.turbulence = turbulence = ml_collections.ConfigDict()
    turbulence.turb_intensity = 1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    turbulence.is_divergence_free = False
    
    solver = config.solver
    solver.max_init_gray_scale = 1.05
    solver.type = 'ade'
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.min_fwd_steps = 1
    solver.max_fwd_steps = solver.n_denoising_steps = 50
    solver.hash = hash_solver(solver)

    debug = False
    if debug:
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        config.training.batch_size = 16
        config.eval.batch_size = 16
        training.n_iters = 5001
        training.sampling_freq = 100


    return config
