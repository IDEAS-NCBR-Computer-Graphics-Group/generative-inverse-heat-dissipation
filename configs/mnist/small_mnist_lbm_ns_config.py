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
    
    # model.blur_sigma_max = 20
    # model.blur_sigma_min = 0.5
    # model.K = 50
    # model.blur_schedule = np.exp(np.linspace(np.log(model.blur_sigma_min),
    #                                          np.log(model.blur_sigma_max), model.K))
    # model.blur_schedule = np.array([0] + list(model.blur_schedule))  # Add the k=0 timestep
    
    data = config.data
    data.showcase_comparison = True
    data.process_pairs = True
    data.process_all = True
    data.processed_filename = 'lbm_ns_pairs' if config.data.process_pairs else 'lbm_ns'
    data.dataset = 'MNIST'

    data.transform = transforms.Compose([])
    
    training = config.training
    training.n_iters = 10001
    training.snapshot_freq = 1000
    training.snapshot_freq_for_preemption = 1000
    training.log_freq = 100
    training.eval_freq = 200
    training.sampling_freq = 200
    
    # turbulence
    turbulence = config.turbulence
    turbulence.turb_intensity = 0 #*1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))
    
    
    solver = config.solver
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.type = 'ns'
    # solver.niu = solver.bulk_visc = 0.5 * 1/6
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 20
    solver.max_fwd_steps = solver.n_denoising_steps + 1  # corruption_amount = np.random.randint(self.min_steps, self.max_steps) we need to add +1 as max_fwd_steps is excluded from tossing
   
    niu_sched  = conf_utils.lin_schedule(0.5 * 1/6, 0.5 * 1/6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)
    
    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    
    debug = True
    if debug:
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        config.training.batch_size = 16
        config.eval.batch_size = 16
        training.n_iters = 5001
        training.sampling_freq = 100



    return config
