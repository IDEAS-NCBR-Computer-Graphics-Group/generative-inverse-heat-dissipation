import unittest

from timeit import default_timer as timer
import os, shutil, sys
from pathlib import Path
import logging

import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scripts import datasets as ihd_datasets
from scripts import sampling, utils
from numerical_solvers.corruptors.CorruptedDatasetCreator import AVAILABLE_CORRUPTORS

from train_corrupted import train
# sys.path.append('pi-inr/')  # PINNFramework etc.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

class TestTrainCorrupted(unittest.TestCase):
    def setUp(self):
        # Initial logging setup
        logging.basicConfig(level=logging.INFO)
        
        # Get the absolute path of the current script
        current_file_path = Path(__file__).resolve()
        self.project_dir = current_file_path.parents[1]

        # Construct the config path relative to the script directory
        config_dir = os.path.join(self.project_dir, "tests", "configs")
        config_file = "ffhq_128_lbm_ns_example.py"
        
        # config_dir = os.path.join(self.project_dir, "configs", "campaign_ffhq_ns_128_v2")
        # config_file = "config_00e20971.py"
                
        self.config_path = os.path.join(config_dir, config_file)
        logging.info(f"config_path = {self.config_path}")

    def test_call_train_corrupted(self):
        train(self.config_path)

if __name__ == '__main__':
    unittest.main()