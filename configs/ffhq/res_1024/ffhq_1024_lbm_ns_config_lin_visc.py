
from configs.ffhq.res_1024 import default_lbm_ns_ffhq_1024_2_1024_config as default_lbm_ffhq_config
import numpy as np
from configs import conf_utils


def get_config():
    config = default_lbm_ffhq_config.get_config()

    turbulence = config.turbulence
    solver = config.solver

    niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)

    stamp = config.stamp
    stamp.hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])

    return config