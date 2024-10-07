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
    min_init_gray_scale: float = 0.95
    max_init_gray_scale: float = 1.05
    
    process_pairs: bool = True
    processed_filename: str = field(init=False) # To be computed in post_init
    
    def __post_init__(self):
        self.processed_filename = 'lbm_ns_pairs' if self.process_pairs else 'lbm_ns'


@dataclass
class SolverConfig:
    niu: float = 0.5 * 1/6
    bulk_visc: float = 0.5 * 1/6

    domain_size: Tuple[float, float] = (1.0, 1.0)
    turb_intensity: float = 0*1E-4
    noise_limiter: Tuple[float, float] = (-1E-3, 1E-3)
    dt_turb: float = 5 * 1E-4
    k_min: float = field(init=False)  # To be computed after domain_size is set
    k_max: float = field(init=False)  # To be computed after domain_size is set
    energy_spectrum: Callable[[float], float] = field(init=False)  # Function field


    min_steps: int = 1
    max_steps: int = 20
    n_denoising_steps: int = 20 # its same as max_steps, but lets experiment with less blurred input for a while
    
    def __post_init__(self):
        # Calculate k_min and k_max based on domain_size
        self.k_min = 2.0 * np.pi / min(self.domain_size)
        self.k_max = 2.0 * np.pi / (min(self.domain_size) / 1024)

        # Define energy_spectrum as a lambda function
        self.energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))

@dataclass
class LBMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)  # Fixed: was incorrectly set as DataConfig


# Function to convert the dataclass configuration to ConfigDict
def get_config() -> ConfigDict:
    # Create an instance of the typed LBMConfig class
    config = LBMConfig()

    # Convert nested dataclasses to nested ConfigDicts
    config_dict = ConfigDict()
    config_dict.data = ConfigDict(vars(config.data))
    config_dict.solver = ConfigDict(vars(config.solver))

    # Manually handle the energy_spectrum function
    config_dict.solver['energy_spectrum'] = config.solver.energy_spectrum

    return config_dict
