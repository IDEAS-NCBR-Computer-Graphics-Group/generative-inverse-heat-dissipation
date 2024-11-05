import unittest

import os, shutil, sys
from pathlib import Path
import logging

from sample_corruption import produce_fwd_sample

# sys.path.append('pi-inr/')  # PINNFramework etc.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

class TestFwdCorruptor(unittest.TestCase):
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


    def test_produce_fwd_sample(self):
        produce_fwd_sample(self.config_path)

if __name__ == '__main__':
    unittest.main()
