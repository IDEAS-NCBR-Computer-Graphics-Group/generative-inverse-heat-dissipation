import numpy as np
from configs import conf_utils

def calc_diff(x, y):
    return np.sqrt(x**2 - y**2)

def calc_Fo(sigma, L):
    Fo = sigma / np.power(L, 2)
    return Fo

def get_timesteps_from_sigma(diffusivity, sigma):
    # sigma = np.sqrt(2 * diffusivity * tc)
    tc = np.power(sigma, 2)/(2*diffusivity)
    return int(tc)

def get_sigma_from_Fo(Fo, L):
    sigma = Fo * np.power(L, 2)
    return sigma

def get_timesteps_from_Fo_niu_L(Fo, diffusivity, L):
    sigma = get_sigma_from_Fo(Fo, L)
    tc = get_timesteps_from_sigma(diffusivity,sigma)
    return int(tc)

def get_Fo_from_tc_niu_L(tc, diffusivity, L):
    sigma = np.sqrt(2. * diffusivity * tc)
    Fo = calc_Fo(sigma, L)
    return Fo

def calculate_t_niu_array_from_0(Fo, niu_min, niu_max, L):
  dt_values, niu_values = [], []
  realizable_dFo, niu_realizable_values= [], []

  for i in range(0, len(Fo)):
    dt = 1
    while True:
      niu = ((L*L*Fo[i]) ** 2)/ (2 * dt)
      if niu <= niu_max:
        dt_values.append(dt)
        niu_values.append(niu)

        if niu < niu_min:
          dFo_step_realizable = np.sqrt(2*dt*niu_min)/np.power(L, 2)
          niu_realizable_values.append(niu_min)
        else:
          dFo_step_realizable = get_Fo_from_tc_niu_L(dt, niu, L)
          niu_realizable_values.append(niu)

        realizable_dFo.append(dFo_step_realizable)
        break
      else: dt += 1
      if dt > 1e+10: break

  return np.array(dt_values, dtype=int), np.array(niu_values), np.array(niu_realizable_values), np.array(realizable_dFo)

def get_ihd_solver_setup(config):
    bs = config.model.blur_schedule
    diff_blur_schedule = [calc_diff(bs[i], bs[i-1]) for  i in range(1, len(bs))]
    
    Fo = calc_Fo(
      diff_blur_schedule, config.data.image_size)
    dt, niu, *_ = calculate_t_niu_array_from_0(
      Fo, config.solver.min_niu, config.solver.max_niu, config.data.image_size)
    
    corrupt_sched =  np.array(list(np.cumsum(dt)), dtype=int)
    config.solver.max_fwd_steps = config.model.K
    config.solver.n_denoising_steps = config.solver.max_fwd_steps - 1
    config.solver.corrupt_sched = corrupt_sched

    config.solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, corrupt_sched[-1], dtype=np.float32)
    config.solver.niu = config.solver_bulk_visc = np.array(sum([[niu[i]]*dt[i] for i in range(len(niu))], []))
    config.solver.final_lbm_step = int(corrupt_sched[-1])

    config.solver.hash = conf_utils.hash_solver(config.solver)

    return config
