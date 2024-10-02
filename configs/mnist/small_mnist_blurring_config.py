from configs.mnist import default_mnist_configs
import ml_collections
import numpy as np
import torch

def get_config():
    config = default_mnist_configs.get_default_configs()
    
    model = config.model
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
    data.min_init_gray_scale = 0.0
    data.max_init_gray_scale = 1.0
    data.process_pairs = True
    data.processed_filename = 'lbm_ns_turb_pairs' if config.data.process_pairs else 'lbm_ns_turb'
    data.dataset = 'CORRUPTED_BLURR_MNIST'

    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100
    
    solver = config.solver 
    solver.type = 'blurr'
    solver.min_steps = 1
    solver.max_steps = 50
    solver.n_denoising_steps = 10

    return config
