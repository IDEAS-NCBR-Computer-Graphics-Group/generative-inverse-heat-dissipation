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
    # min_init_gray_scale: float = 0.95
    # max_init_gray_scale: float = 1.05    
    min_init_gray_scale: float = 0.0
    max_init_gray_scale: float = 1.0
    
    process_pairs: bool = True
    
    def __post_init__(self):
        self.processed_filename = 'blurr_pairs' if self.process_pairs else 'blurr'


@dataclass
class SolverConfig:
    step_size: float = 0.1
    min_steps: float = field(init=False)  # To be computed after domain_size is set
    max_steps: float = 10. # max amount of blurr

    def __post_init__(self):
        self.min_steps: float = self.step_size # min amount of blurr

    
@dataclass
class BlurrConfig:
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)  # Fixed: was incorrectly set as DataConfig

# Function to convert the dataclass configuration to ConfigDict
def get_config() -> ConfigDict:
    # Create an instance of the typed LBMConfig class
    config = BlurrConfig()

    # Convert nested dataclasses to nested ConfigDicts
    config_dict = ConfigDict()
    config_dict.data = ConfigDict(vars(config.data))
    config_dict.solver = ConfigDict(vars(config.solver))

    return config_dict
