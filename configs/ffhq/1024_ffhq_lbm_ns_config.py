import torch
import ml_collections
from configs.ffhq import default_lbm_ffhq_config
from configs.conf_utils import hash_solver
from torchvision import transforms

def get_config():
    config = default_lbm_ffhq_config.get_default_configs()
    training = config.training 
    config.training.batch_size = 1
    training.n_iters = 1001 # 1300001
    training.snapshot_freq = 100 #50000
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 100 #10000
    
    evaluate = config.eval
    evaluate.batch_size = 1
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.calculate_fids = False

    model = config.model
    model.model_channels = 32 
    
    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.dataset = 'FFHQ'
    data.processed_filename = 'lbm_ns_pairs' if data.process_pairs else 'lbm_ns'
    data.image_size = 256
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1
    
    data.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Grayscale()
                                        ])

    turbulence = config.turbulence
    turbulence.turb_intensity = 0 #*1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
    turbulence.is_divergence_free = False
    
    solver = config.solver
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.type = 'ns'
    solver.niu = solver.bulk_visc = 0.5 * 1/6
    solver.min_fwd_steps = 1
    solver.max_fwd_steps = solver.n_denoising_steps = 20
    solver.hash = hash_solver(solver)

    debug = True
    if debug:
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        config.training.batch_size = 1
        config.eval.batch_size = 16
        training.n_iters = 5001
        training.sampling_freq = 100


    return config
