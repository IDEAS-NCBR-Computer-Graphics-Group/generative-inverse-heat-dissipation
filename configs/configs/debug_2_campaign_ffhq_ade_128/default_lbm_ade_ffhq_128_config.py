import ml_collections
import torch
import numpy as np
from configs import conf_utils
from torchvision import transforms
import os

from configs.ffhq.ihd.default_ffhq_configs import get_config as get_config_ihd
from configs.match_sim_numbers import get_ihd_solver_setup, calculate_u_max

def get_config():
    return get_default_configs()

def get_default_configs():
    config = get_config_ihd()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.n_evals = 25 # batches for test-set evaluation, arbitrary choice
    training.n_iters = 1300001
    training.log_freq = 1000
    training.eval_freq = 2000
    training.sampling_freq = 10000
    
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq = 50000
    training.snapshot_freq_for_preemption = 10000

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
    data.num_channels = 3

    # data - cd
    data = config.data
    data.showcase_comparison = True
    data.process_all = True
    data.process_pairs = True
    data.processed_filename = 'lbm_ade_pairs' if data.process_pairs else 'lbm_ade'
    data.dataset = 'FFHQ_128'
    data.image_size = 128
    data.transform = transforms.Compose(
        [transforms.ToTensor()])

    # solver
    solver = config.solver =  ml_collections.ConfigDict()

    solver.type = 'ade'
    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.min_fwd_steps = 1
    solver.n_denoising_steps = 200
    solver.max_fwd_steps = solver.n_denoising_steps + 1  # corruption_amount = np.random.randint(self.min_steps, self.max_steps) thus we need to add +1 as max_fwd_steps is excluded from tossing
    solver.final_lbm_step = 10000
    solver.min_niu = 1E-4* 1./ 6
    solver.max_niu = 1./ 6
    solver.max_cs2 = solver.min_cs2 = 1./3


    # solver.corrupt_sched = conf_utils.lin_schedul(
    #         solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
    
    solver.corrupt_sched = conf_utils.exp_schedule(
        solver.min_fwd_steps, solver.final_lbm_step, solver.max_fwd_steps, dtype=int)
        
    are_steps_unique = False
    if are_steps_unique:
        solver.corrupt_sched = np.unique(solver.corrupt_sched)
        solver.max_fwd_steps = len(solver.corrupt_sched)
        solver.n_denoising_steps = solver.max_fwd_steps - 1
    
    solver.cs2 = conf_utils.lin_schedule(solver.min_cs2, solver.max_cs2, solver.final_lbm_step, dtype=np.float32)
    
    # niu_sched = conf_utils.lin_schedule(1E-4*1 / 6, 1E-4 * 1 / 6, solver.final_lbm_step, dtype=np.float32)
    niu_sched = conf_utils.tanh_schedule(solver.min_niu, solver.max_niu, solver.final_lbm_step, dtype=np.float32)
    # niu_sched  = conf_utils.exp_schedule(1E-4 * 1./6., 1./6., solver.max_fwd_steps)

    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)

    config = get_ihd_solver_setup(config)

    # turbulence
    turbulence = config.turbulence = ml_collections.ConfigDict()
    turbulence.max_turb_intensity = 0 * 1E-6
    turbulence.min_turb_intensity = 0 * 5E-4
    turbulence.turb_intensity = conf_utils.lin_schedule(turbulence.min_turb_intensity, turbulence.max_turb_intensity, solver.final_lbm_step, dtype=np.float32)
    turbulence.noise_limiter = (-1E-2, 1E-2)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 1 * 1E-3
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0, k ** (turbulence.energy_slope))

    config.turb_intensity = calculate_u_max(niu_sched, Pe = 100, L = data.image_size)

    # model
    config.model = model = ml_collections.ConfigDict()
    
    model.sigma = 0.01
    model.dropout = 0.3
    model.model_channels = 128
    model.channel_mult = (1, 2, 3, 4, 5)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = (2, 3, 4)
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.num_res_blocks = 3
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True
    model.hash = conf_utils.hash_solver(model)

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    optim.automatic_mp = True
    optim.hash = conf_utils.hash_solver(optim)

    config.stamp = stamp = ml_collections.ConfigDict()
    stamp = config.stamp
    stamp.fwd_solver_hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])
    stamp.model_optim_hash = conf_utils.hash_joiner([model.hash, optim.hash, training.hash])

    config.seed = 42
    config.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    if os.uname().nodename in ['pc', 'laptop', 'armadillo']:
        debug = True
    else:
        debug = False
    
    # debug = True
    if debug:
        data = config.data
        data.processed_filename = f'{data.processed_filename}_debug'
        data.process_all = False
        
        model = config.model
        model.model_channels = 32
        model.channel_mult = (1, 1, 2, 2, 2)
        model.attention_levels = (3, 4)

        
        config.training.n_iters = 51
        config.training.batch_size = 1  # rtx2080
        config.eval.batch_size = 16
        config.training.sampling_freq = 25
        config.training.log_freq = 10
        config.training.eval_freq = 20
        
    return config
