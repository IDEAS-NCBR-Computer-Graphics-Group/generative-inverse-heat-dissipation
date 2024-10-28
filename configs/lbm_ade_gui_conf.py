
import ml_collections
import numpy as np
import torch
from configs import conf_utils


def get_config():
    config = ml_collections.ConfigDict()
    # solver
    config.solver = solver = ml_collections.ConfigDict()
    config.turbulence = turbulence = ml_collections.ConfigDict()
    config.stamp = stamp = ml_collections.ConfigDict()

    # turbulence
    turbulence = config.turbulence
    turbulence.turb_intensity = 1* 1E-3
    turbulence.noise_limiter = (-1E-3, 1E-3)
    turbulence.domain_size = (1.0, 1.0)
    turbulence.dt_turb = 5 * 1E-4
    turbulence.k_min = 2.0 * torch.pi / min(turbulence.domain_size)
    turbulence.k_max = 2.0 * torch.pi / (min(turbulence.domain_size) / 1024)
    turbulence.is_divergence_free = False
    turbulence.energy_slope = -5.0 / 3.0
    turbulence.hash = conf_utils.hash_solver(turbulence)
    turbulence.energy_spectrum = lambda k: torch.where(torch.isinf(k ** (turbulence.energy_slope)), 0,
                                                       k ** (turbulence.energy_slope))

    solver = config.solver

    solver.min_init_gray_scale = 0.95
    solver.max_init_gray_scale = 1.05
    solver.type = 'ade'

    solver.cs2 = 1./3
    solver.max_fwd_steps = 5000
    niu_sched = conf_utils.lin_schedule(0.001 * 1 / 6, 0.5 * 1 / 6, solver.max_fwd_steps)

    # niu_sched  = conf_utils.exp_schedule(1E-4 * 1./6., 1./6., solver.max_fwd_steps)
    # niu_sched = conf_utils.inv_cosine_aplha_schedule(1E-4 * 1./6., 1./6., solver.max_fwd_steps)
    # niu_sched = conf_utils.tanh_schedule(1E-4 * 1. / 6., 1. / 6., solver.max_fwd_steps)
    # niu_sched  = lin_schedule(1E-4 * 1./6., 1./6., solver.max_fwd_steps)
    # niu_sched  = lin_schedule(1E-4 * 1./6., 1E-4 *1./6., solver.max_fwd_steps)

    solver.niu = solver.bulk_visc = niu_sched

    solver.hash = conf_utils.hash_solver(solver)

    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])


    return config
