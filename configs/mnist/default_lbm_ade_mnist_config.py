import ml_collections
import torch
from configs import conf_utils
from torchvision import transforms
import os

from configs.mnist.ihd.default_mnist_configs import get_config as get_config_ihd

from configs import match_sim_numbers

def get_config():
    return get_default_configs()

def get_default_configs():
    config = get_config_ihd()

    # training
    training = config.training
    training.log_freq = 1000
    training.eval_freq = 2000
    training.sampling_freq = 10000
    training.snapshot_freq = 50000
    training.snapshot_freq_for_preemption = 10000
    training.hash = conf_utils.hash_int(config.training.batch_size)

    # data
    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ade_pairs' if data.process_pairs else 'lbm_ade'
    data.dataset = 'MNIST'
    data.image_size = 28
    data.transform = transforms.Compose([])

    # solver
    solver = config.solver =  ml_collections.ConfigDict()
    solver.type = 'ade'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.min_fwd_steps = 1
    solver.max_fwd_steps = solver.n_denoising_steps = 200
    solver.final_lbm_step = 100
    solver.min_niu = 1E-4* 1./ 6
    solver.max_niu = 1./ 6
    solver.max_cs2 = solver.min_cs2 = 1./3
    solver = match_sim_numbers.get_ihd_solver_setup(
        config.model.K, config.model.blur_schedule, data.image_size, solver)

    # turbulence
    turbulence = config.turbulence = ml_collections.ConfigDict()
    turbulence.Pe = 0.0
    turbulence.turb_intensity = match_sim_numbers.u_from_Pe(turbulence.Pe, solver.niu, data.image_size)
    turbulence.noise_limiter = (-1E-2, 1E-2)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 1 * 1E-3
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))

    # model
    model = config.model
    model.hash = conf_utils.hash_solver(model)

    # optimization
    optim = config.optim
    optim.automatic_mp = True
    optim.hash = conf_utils.hash_solver(optim)

    config.stamp = stamp = ml_collections.ConfigDict()
    stamp = config.stamp
    stamp.fwd_solver_hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([model.hash, optim.hash, training.hash])

    if os.uname().nodename in ['pc', 'laptop', 'armadillo']: debug = True
    else: debug = False
    
    if debug:
        data = config.data
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        
        config.training.n_iters = 10001 #51
        config.training.batch_size = 16
        config.eval.batch_size = 16
        config.training.sampling_freq = 100 #25
        config.training.log_freq = 50 #10
        config.training.eval_freq = 50 #20
        
    return config
