# config.py
import numpy as np

from ml_collections import ConfigDict
from dataclasses import dataclass


from ml_collections import ConfigDict
from dataclasses import dataclass, field
from typing import Tuple, Callable
import numpy as np

# Define the nested configurations as dataclasses
@dataclass
class DataConfig:
    image_size: int = 28  # for MNIST

@dataclass
class SolverConfig:
    niu: float = 0.5 * 1/6
    bulk_visc: float = 0.5 * 1/6
    
    domain_size: Tuple[float, float] = (1.0, 1.0)
    turb_intensity: float = 1E-4
    noise_limiter: Tuple[float, float] = (-1E-3, 1E-3)
    dt_turb: float = 5 * 1E-4
    k_min: float = field(init=False)  # To be computed after domain_size is set
    k_max: float = field(init=False)  # To be computed after domain_size is set
    energy_spectrum: Callable[[float], float] = field(init=False)  # Function field


    min_lbm_steps: int = 2
    max_lbm_steps: int = 50
    
    
    def __post_init__(self):
        # Calculate k_min and k_max based on domain_size
        self.k_min = 2.0 * np.pi / min(self.domain_size)
        self.k_max = 2.0 * np.pi / (min(self.domain_size) / 1024)

        # Define energy_spectrum as a lambda function
        self.energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))

@dataclass
class LBMConfig:
    data: DataConfig = DataConfig()
    solver: SolverConfig = SolverConfig()

# Function to convert the dataclass configuration to ConfigDict
def get_lbm_ns_config() -> ConfigDict:
    # Create an instance of the typed LBMConfig class
    config = LBMConfig()

    # Convert nested dataclasses to nested ConfigDicts
    config_dict = ConfigDict()
    config_dict.data = ConfigDict(vars(config.data))
    config_dict.solver = ConfigDict(vars(config.solver))

    return config_dict




# @dataclass
# class Config:
#     learning_rate: float = 0.001
#     batch_size: int = 32
#     num_epochs: int = 10
#     model_name: str = 'resnet50'


# def get_lbm_ns_config() -> ConfigDict:
#     # cfd solver
#     config = ConfigDict()
    
#     # data
#     config.data = data = ConfigDict()
#     data.image_size = 28 # mnist
    
#     config.solver = solver = ConfigDict()
#     solver = ConfigDict()
#     solver.niu = 0.5 * 1/6
#     solver.bulk_visc = 0.5 * 1/6
#     solver.domain_size = (1.0, 1.0)

#     solver.turb_intensity = 1E-4
#     solver.noise_limiter = (-1E-3, 1E-3)
#     solver.dt_turb = 5 * 1E-4
#     solver.k_min = 2.0 * np.pi / min(solver.domain_size)
#     solver.k_max = 2.0 * np.pi / (min(solver.domain_size) / 1024)
    
#     solver.energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
#     return solver
