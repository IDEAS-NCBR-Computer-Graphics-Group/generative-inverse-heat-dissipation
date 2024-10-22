from configs.mnist import default_lbm_mnist_config as default_mnist_configs
import ml_collections
import numpy as np
import torch
from torchvision import transforms
from configs import conf_utils

def get_config():
    config = default_mnist_configs.get_default_configs()
    
    model = config.model
    model.model_channels = 64
    model.channel_mult = (1, 1, 1)
    
    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.process_all = True
    data.processed_filename = 'lbm_ade_turb_pairs' if config.data.process_pairs else 'lbm_ade_turb'
    data.dataset = 'MNIST'
    data.transform = transforms.Compose([])
    
    training = config.training
    training.n_iters = 1001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 100
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100

    turbulence = config.turbulence
    turbulence.turb_intensity =  0*1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))
    
    
    # Define the solver configuration
    solver = config.solver = ml_collections.ConfigDict()  # Create solver attribute
    solver.type = 'ade'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    # solver.niu = solver.bulk_visc =  0.5 * 1/6
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 50
    solver.max_fwd_steps = solver.n_denoising_steps + 1  # corruption_amount = np.random.randint(self.min_steps, self.max_steps) we need to add +1 as max_fwd_steps is excluded from tossing
   
    niu_sched  = conf_utils.lin_schedule(0.5 * 1/6, 0.5 * 1/6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)
    
    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    
    debug = False
    if debug:
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        config.training.batch_size = 16
        config.eval.batch_size = 16
        training.n_iters = 5001
        training.sampling_freq = 100

    return config