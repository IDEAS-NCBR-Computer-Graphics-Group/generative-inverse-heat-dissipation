
from configs.ffhq.res_512 import default_lbm_ns_ffhq_config
import numpy as np
from configs import conf_utils


def get_config():
    config = default_lbm_ns_ffhq_config.get_config()

    data = config.data
    data.image_size = 256
    data.corrupted_image_size_write = 256
    
    turbulence = config.turbulence
    solver = config.solver

    niu_sched = conf_utils.lin_schedule(1E-4 * 1 / 6, 1 / 6, solver.max_fwd_steps)
    solver.niu = solver.bulk_visc = niu_sched
    solver.hash = conf_utils.hash_solver(solver)

    stamp = config.stamp
    stamp.fwd_solver_hash = conf_utils.hash_joiner([solver.hash, turbulence.hash])

    return config