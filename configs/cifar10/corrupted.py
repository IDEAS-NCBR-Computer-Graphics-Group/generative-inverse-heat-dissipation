from configs.cifar10 import default_cifar10_configs
import numpy as np


def get_config():
    return get_default_configs()


def get_default_configs():
    config = default_cifar10_configs.get_config()

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
    
    training = config.training
    training.log_freq = 50
    training.eval_freq = 100
    training.n_iters = 20001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.sampling_freq = 100

    return config
