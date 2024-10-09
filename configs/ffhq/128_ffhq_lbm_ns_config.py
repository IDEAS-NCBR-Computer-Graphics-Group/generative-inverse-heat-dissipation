import ml_collections
from configs.ffhq import default_lbm_ffhq_config
import numpy as np
import torch
from torchvision import transforms

def get_config():
    return get_default_configs()

def get_default_configs():
    config = default_lbm_ffhq_config.get_default_configs()

    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ns_pairs' if data.process_pairs else 'lbm_ns'
    data.dataset = 'FFHQ_128'
    
    data.image_size = 128
    data.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Grayscale()
                                         ])
    data.num_channels = 1
    
    training = config.training
    training.n_iters = 1000001
    training.snapshot_freq = 2000
    training.snapshot_freq_for_preemption = 2000
    training.log_freq = 100
    training.eval_freq = 200
    training.sampling_freq = 2000


    turbulence = config.turbulence 
    turbulence.turb_intensity = 0 #*1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    turbulence.is_divergence_free = False
    
    solver = config.solver
    solver.type = 'ns'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.niu = solver.bulk_visc =  0.5 * 1/6
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = solver.max_fwd_steps = 100

    
    debug = False
    if debug:
        data.processed_filename = 'lbm_ns_pairs_debug' if data.process_pairs else 'lbm_ns_debug'
        data.process_all = False
        config.training.batch_size = 4 # rtx2080
        config.eval.batch_size = 4
        # config.training.batch_size = 16 # rtx4080

        training.n_iters = 5001
        training.sampling_freq = 100
        
    return config
    