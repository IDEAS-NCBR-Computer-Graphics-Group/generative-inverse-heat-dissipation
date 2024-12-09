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

def get_ihd_solver_setup(K, blur_schedule, image_size, solver):
    bs = blur_schedule
    diff_blur_schedule = [calc_diff(bs[i], bs[i-1]) for  i in range(1, len(bs))]
    diff_blur_schedule = [bs[0] , *diff_blur_schedule]
    
    Fo = calc_Fo(diff_blur_schedule, image_size)
    dt, niu, *_ = calculate_t_niu_array_from_0(Fo, solver.min_niu, solver.max_niu, image_size)
    
    solver.corrupt_sched =  np.array(list(np.cumsum(dt)), dtype=int)
    solver.n_denoising_steps = solver.max_fwd_steps = K

    solver.cs2 = conf_utils.lin_schedule(1./3, 1./3, solver.corrupt_sched[-1], dtype=np.float32)
    solver.niu = solver.bulk_visc = np.array(sum([[niu[i]]*dt[i] for i in range(len(niu))], []))
    solver.final_lbm_step = int(solver.corrupt_sched[-1])

    solver.hash = conf_utils.hash_solver(solver)
    return solver

def u_from_Pe(Pe, niu, L):
        u = Pe*niu/L
        return u    # Pe = conf_utils.lin_schedule(5E-1, 1E-0, solver.final_lbm_step, dtype=np.float32)