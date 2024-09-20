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
    image_size: int = 128  # for MNIST
    # min_init_gray_scale: float = 0.95
    # max_init_gray_scale: float = 1.05
    
    min_init_gray_scale: float = 0.0
    max_init_gray_scale: float = 1.0
    
    process_pairs: bool = True
    
    def __post_init__(self):
        self.processed_filename = 'blurr_pairs' if self.process_pairs else 'blurr'


@dataclass
class SolverConfig:
    min_blurr: float = 1.
    max_blurr: float = 5.

@dataclass
class BlurrConfig:
    data: DataConfig = field(default_factory=DataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)  # Fixed: was incorrectly set as DataConfig

# Function to convert the dataclass configuration to ConfigDict
def get_blurr_config() -> ConfigDict:
    # Create an instance of the typed LBMConfig class
    config = BlurrConfig()

    # Convert nested dataclasses to nested ConfigDicts
    config_dict = ConfigDict()
    config_dict.data = ConfigDict(vars(config.data))
    config_dict.solver = ConfigDict(vars(config.solver))

    return config_dict
