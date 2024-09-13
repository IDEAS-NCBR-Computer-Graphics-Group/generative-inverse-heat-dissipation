from configs.mnist import default_mnist_configs
import ml_collections
import numpy as np


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
    model.blur_schedule = np.array(
        [0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    config.data.dataset = 'CORRUPTED_MNIST'
    
    config.training.snapshot_freq_for_preemption = 100
    config.training.sampling_freq = 100
    
    # cfd solver
    config.solver = solver = ml_collections.ConfigDict()
    solver.niu = 0.5 * 1/6
    solver.bulk_visc = 0.5 * 1/6
    solver.domain_size = (1.0, 1.0)

    solver.turb_intensity = 1E-4
    solver.noise_limiter = (-1E-3, 1E-3)
    solver.dt_turb = 5 * 1E-4
    solver.k_min = 2.0 * np.pi / min(solver.domain_size)
    solver.k_max = 2.0 * np.pi / (min(solver.domain_size) / 1024)
    
    solver.energy_spectrum= lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    return config
