import ml_collections
import torch
import numpy as np
from configs import conf_utils

def get_config():
    return get_default_configs()


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 1300001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    training.sampling_freq = 1000
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000 
    training.n_evals = 25 # 25 batches for test-set evaluation, arbitrary choice
    
    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 256
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

    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.dataset = 'MNIST'
    data.image_size = 28
    data.processed_filename = 'lbm_ns_pairs' if data.process_pairs else 'lbm_ns'
    data.transform = transforms.Compose(
        [transforms.ToTensor()])

    # solver
    config.solver = solver = ml_collections.ConfigDict()
    config.turbulence = turbulence = ml_collections.ConfigDict()
    config.stamp = stamp = ml_collections.ConfigDict()
    
    # solver
    config.turbulence = turbulence = ml_collections.ConfigDict()
    turbulence.turb_intensity = float(0.) #*1E-4
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))
    
    config.solver = solver = ml_collections.ConfigDict()
    solver.type = 'ns'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.cs2 = 1./3.
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 50
    solver.max_fwd_steps = solver.n_denoising_steps + 1 # corruption_amount = np.random.randint(self.min_steps, self.max_steps) thus we need to add +1 as max_fwd_steps is excluded from tossing
    solver.final_lbm_step = 50
    solver.lin_sched = False

    if solver.lin_sched: 
        solver.corrupt_sched = np.linspace(
            solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
    else:
        solver.lbm_steps_base = 2.0
        solver.starting_lbm_steps_pow = np.emath.logn(solver.lbm_steps_base, solver.min_fwd_steps)
        solver.final_lbm_steps_pow = np.emath.logn(solver.lbm_steps_base, solver.final_lbm_step)
        if np.math.pow(solver.lbm_steps_base, solver.final_lbm_steps_pow) != solver.final_lbm_step:
            solver.final_lbm_steps_pow += 2*np.finfo(float).eps
        solver.corrupt_sched = np.logspace(
            solver.starting_lbm_steps_pow, solver.final_lbm_steps_pow,
            solver.max_fwd_steps, base=solver.lbm_steps_base, dtype=int)

    niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma = 0.01
    model.dropout = 0.1
    model.model_channels = 128  # Base amount of channels in the model
    model.channel_mult = (1, 2, 2)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = (2,)
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.num_res_blocks = 4
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    optim.automatic_mp = False
    optim.hash = conf_utils.hash_solver(optim)
    
    config.seed = 42
    
    # config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
    return config
