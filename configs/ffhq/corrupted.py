from configs.ffhq import default_ffhq_configs
import numpy as np

# Config for the model where image resolution = 128x128, and
# maximum blurring effective length scale is 128 as well
# -> the average colour and other characteristics get disentangled

def get_config():
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
    
    training = config.training
    training.log_freq = 50
    training.eval_freq = 100
    training.n_iters = 20001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.sampling_freq = 100

    return config