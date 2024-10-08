from configs.mnist.ihd import default_mnist_configs
import numpy as np


def get_config():
    config = default_mnist_configs.get_default_configs()
    model = config.model
    
    model.blur_sigma_max = 20
    model.blur_sigma_min = 0.5
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
    model.K = 50
    model.blur_schedule = np.linspace(model.blur_sigma_min, model.blur_sigma_max, model.K)
    model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    
    config.training.n_iters = 10001
    config.training.snapshot_freq = 1000
    config.training.snapshot_freq_for_preemption = 500
    config.training.sampling_freq = 500
    return config