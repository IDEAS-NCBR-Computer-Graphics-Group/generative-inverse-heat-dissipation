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
    

    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.process_all = True 
    data.processed_filename = 'gaussian_blurr_pairs' if config.data.process_pairs else 'gaussian_blurr'
    data.dataset = 'MNIST'
    data.transform = transforms.Compose([])

    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100
    
    solver = config.solver 
    solver.type = 'gaussian_blurr'
    solver.min_init_gray_scale = 0.
    solver.max_init_gray_scale = 1.
    
    
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = solver.max_fwd_steps = 50
    
    solver.blur_sigma_min = 0.5
    solver.blur_sigma_max = 20


    solver.blur_schedule = np.exp(np.linspace(np.log(solver.blur_sigma_min),
                                             np.log(solver.blur_sigma_max), solver.max_fwd_steps))
    solver.blur_schedule = np.array([0] + list(solver.blur_schedule))  # Add the k=0 timestep

    return config
