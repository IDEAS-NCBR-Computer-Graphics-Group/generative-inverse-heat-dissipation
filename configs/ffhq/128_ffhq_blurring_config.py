import ml_collections
from configs.ffhq import default_ffhq_configs
import numpy as np
import torch
from torchvision import transforms

def get_config():
    return get_default_configs()

def get_default_configs():
    config = default_ffhq_configs.get_default_configs()
    config.training.batch_size = 4

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
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'gaussian_blurr_pairs' if data.process_pairs else 'gaussian_blurr'
    data.dataset = 'FFHQ_128'

    data.image_size = 128
    data.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Grayscale()
                                         ])
    data.num_channels = 1

    
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
    solver.n_denoising_steps = solver.max_fwd_steps = 200
    
    model.blur_sigma_max = 128
    model.blur_sigma_min = 0.5

    solver.blur_schedule = np.exp(np.linspace(np.log(solver.blur_sigma_min),
                                             np.log(solver.blur_sigma_max), solver.max_fwd_steps))
    solver.blur_schedule = np.array([0] + list(solver.blur_schedule))  # Add the k=0 timestep

 

    optim = config.optim
    optim.automatic_mp = False
    

    return config
    