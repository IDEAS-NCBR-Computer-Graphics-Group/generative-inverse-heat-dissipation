import torch
import numpy as np
from configs.ffhq.ihd import default_ffhq_configs

def get_config():
    config = default_ffhq_configs.get_default_configs()
    training = config.training 
    config.training.batch_size = 1
    training.n_iters = 1001 # 1300001
    training.snapshot_freq = 100 #50000
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100 #10000
    
    model = config.model
    model.model_channels = 64 
    
    return config
