import ml_collections
from configs.ffhq import default_lbm_ffhq_config
import numpy as np
import torch
from torchvision import transforms
from configs import conf_utils


def get_config():
    return get_default_configs()

def get_default_configs():
    config = default_lbm_ffhq_config.get_default_configs()

    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ns_pairs' if data.process_pairs else 'lbm_ns'
    data.dataset = 'FFHQ_128'
    
    data.image_size = 128
    data.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Grayscale()
                                         ])
    data.num_channels = 1
    
    training = config.training
    training.n_iters = 100001
    training.snapshot_freq = 2000
    training.snapshot_freq_for_preemption = 2000
    training.log_freq = 100
    training.eval_freq = 200
    training.sampling_freq = 2000

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
    solver.type = 'ns'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05

    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 100
    solver.max_fwd_steps = solver.n_denoising_steps + 1  # corruption_amount = np.random.randint(self.min_steps, self.max_steps) thus we need to add +1 as max_fwd_steps is excluded from tossing
    
    # niu_sched  = conf_utils.exp_schedule(1E-4 * 1./6., 1./6., n)
    niu_sched  = conf_utils.lin_schedule(0.5 * 1/6, 0.5 * 1/6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc =  niu_sched
    solver.hash = conf_utils.hash_solver(solver)
    
    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    
    debug = False
    if debug:
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        
        model = config.model
        model.channel_mult = (1, 2, 3, 3, 3)
        # model.attention_levels = (2, 3, 4)
    
        config.training.batch_size = 4 # rtx2080
        config.eval.batch_size = 4
        # config.training.batch_size = 16 # rtx4080

        training.n_iters = 1001
        training.sampling_freq = 200
        
    return config
    