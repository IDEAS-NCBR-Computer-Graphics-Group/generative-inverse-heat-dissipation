from configs.ffhq.ihd import small_img_size_128
import numpy as np

# Config for the model where image resolution = 128x128, and
# maximum blurring effective length scale is 128 as well
# -> the average colour and other characteristics get disentangled

def get_config():
    config = small_img_size_128.get_config()
    model = config.model
    
    model.K = 600 # was 200
    model.blur_schedule = np.linspace(model.blur_sigma_min, model.blur_sigma_max, model.K)
    model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    return config

