from configs.ffhq import default_ffhq_configs
import numpy as np

# Config for the model where image resolution = 128x128, and
# maximum blurring effective length scale is 128 as well
# -> the average colour and other characteristics get disentangled

def get_config():
    config = default_ffhq_configs.get_default_configs()
    model = config.model
    config.training.batch_size = 4
    config.data.image_size = 128
    config.data.dataset = 'FFHQ_128' 
    config.training.n_iters = 1001

    config.training.snapshot_freq = 1000
    config.training.snapshot_freq_for_preemption = 100
    config.training.sampling_freq = 100
    
    return config