from configs.mnist import default_mnist_configs
import ml_collections
import numpy as np
import torch
from torchvision import transforms

def get_config():
    config = default_mnist_configs.get_default_configs()
    
    model = config.model
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
    
    model.blur_sigma_max = 20
    model.blur_sigma_min = 0.5
    model.K = 50
    model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
                                             np.log(model.blur_sigma_max), model.K))
    model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ns_turb_pairs' if config.data.process_pairs else 'lbm_ns_turb'
    data.dataset = 'MNIST'
    data.min_init_gray_scale = 0.95
    data.max_init_gray_scale = 1.05
    data.transform = transforms.Compose([])
    
    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100
    
    solver = config.solver
    solver.type = 'NS'
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.domain_size = (1.0, 1.0)
    
    solver.turb_intensity = 0*1E-4
    solver.noise_limiter = (-1E-3, 1E-3)
    solver.dt_turb = 5 * 1E-4
    solver.k_min = 2.0 * torch.pi / min(solver.domain_size)
    solver.k_max = 2.0 * torch.pi / (min(solver.domain_size) / 1024)
    solver.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    solver.min_steps = 1
    solver.max_steps = 20
    solver.is_divergence_free = False
    
    solver.n_denoising_steps = 20

    return config
