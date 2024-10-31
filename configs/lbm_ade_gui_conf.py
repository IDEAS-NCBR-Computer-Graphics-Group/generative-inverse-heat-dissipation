
import ml_collections
import numpy as np
import torch
from torchvision import transforms

from configs import conf_utils

def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.n_evals = 25  # batches for test-set evaluation, arbitrary choice
    training.n_iters = 25001  # 1300001
    training.log_freq = 100
    training.eval_freq = 200
    training.sampling_freq = 2000  # 10000

    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq = 5000  # 50000
    training.snapshot_freq_for_preemption = 5000

    training.hash = conf_utils.hash_int(config.training.batch_size)

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 4
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.calculate_fids = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1

    # data - cd
    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ns_pairs' if data.process_pairs else 'lbm_ns'
    data.dataset = 'FFHQ_128'
    data.image_size = 128
    data.transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale()])

    # solver
    config.solver = solver = ml_collections.ConfigDict()
    config.turbulence = turbulence = ml_collections.ConfigDict()
    config.stamp = stamp = ml_collections.ConfigDict()

    solver = config.solver

    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05

    solver.type = 'ade'

    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 200
    solver.max_fwd_steps = solver.n_denoising_steps + 1  # corruption_amount = np.random.randint(self.min_steps, self.max_steps) thus we need to add +1 as max_fwd_steps is excluded from tossing
    solver.final_lbm_step = 500

    # solver.corrupt_sched = conf_utils.lin_schedul(
    #         solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
    
    solver.corrupt_sched = conf_utils.exp_schedule(
        solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
        
    are_steps_unique = False
    if are_steps_unique:
        solver.corrupt_sched = np.unique(solver.corrupt_sched)
        solver.max_fwd_steps = len(solver.corrupt_sched)
        solver.n_denoising_steps = solver.max_fwd_steps - 1
    
    solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, solver.final_lbm_step, dtype=np.float32)
    # niu_sched = conf_utils.lin_schedule(1E-4*1 / 6, 1E-4 * 1 / 6, solver.final_lbm_step, dtype=np.float32)
    niu_sched = conf_utils.tanh_schedule(1E-4* 1./ 6,  1./ 6, solver.final_lbm_step, dtype=np.float32)
    # niu_sched  = conf_utils.exp_schedule(1E-4 * 1./6., 1./6., solver.max_fwd_steps)

    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)

    
    # turbulence
    turbulence = config.turbulence
    turbulence.turb_intensity = conf_utils.lin_schedule(1E-6, 5E-4, solver.final_lbm_step, dtype=np.float32)
    turbulence.noise_limiter = (-1E-2, 1E-2)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))

    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])


    return config
