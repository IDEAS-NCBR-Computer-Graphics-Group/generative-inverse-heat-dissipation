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
        self.config_dir = os.path.join(self.project_dir, "tests", "configs")
        self.config_files = ["ffhq_128_lbm_ade_example.py","ffhq_128_lbm_ns_example.py", "ffhq_128_lbm_ns_with_u_example.py"]
        
    def test_produce_fwd_sample(self):
        # test your campaign before launch
        # self.config_dir = os.path.join(self.project_dir, "configs", "campaign_ffhq_ade_128")
        # self.config_files = ["config_d17dedb5.py"]
        
        for config_file in self.config_files:
            config_path = os.path.join(self.config_dir, config_file)
            logging.info(f"fwd corruption: config_path = {config_path}")
            produce_fwd_sample(config_path)

if __name__ == '__main__':
    unittest.main()
