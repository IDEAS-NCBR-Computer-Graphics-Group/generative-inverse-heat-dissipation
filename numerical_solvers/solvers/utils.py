import numpy as np

def exp_schedule(min_value, max_value, n):
    return np.exp(np.linspace(np.log(min_value), np.log(max_value), n))

def lin_schedule(min_value, max_value, n):
    return np.linspace(min_value ,max_value, n)