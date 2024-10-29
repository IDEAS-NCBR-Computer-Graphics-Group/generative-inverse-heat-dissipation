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
        config_dir = os.path.join(self.project_dir, "configs", "ffhq")
        config_file = "ffhq_128_lbm_ns_example.py"
        
        # config_dir = os.path.join(self.project_dir, "configs", "campaign_ffhq_ns_128_v2")
        # config_file = "config_00e20971.py"
                
        self.config_path = os.path.join(config_dir, config_file)
        logging.info(f"config_path = {self.config_path}")


    def test_produce_fwd_sample(self):
        config = utils.load_config_from_path(self.config_path)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        save_dir = os.path.join("tests", "artifacts", f"test_corruption_{config.stamp.fwd_solver_hash}")
        utils.setup_logging(save_dir)
        logging.info(f"save_dir: {save_dir}")
        logging.info(f"config.corrupt_sched has {len(config.solver.corrupt_sched)} elements:\n {config.solver.corrupt_sched}")

        # remove previous corrupted dataset
        dataset_dir = utils.get_save_dir(self.project_dir, config)
        logging.info(f"dataset_dir: {dataset_dir}")
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            logging.info(f"Removed {dataset_dir}")
            
        trainloader, testloader = ihd_datasets.get_dataset(config,
                                                           uniform_dequantization=config.data.uniform_dequantization)

        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(self.config_path, save_dir)
        
        default_cfg_path = os.path.join(*self.config_path.split(os.sep)[0:2], f'default_lbm_{config.data.dataset.lower()}_config.py')     
        if os.path.isfile(default_cfg_path):
            shutil.copy2(default_cfg_path, save_dir)

        clean_image, batch = ihd_datasets.prepare_batch(iter(trainloader), 'cpu')
        corrupted_image, less_corrupted_image, corruption_amount, label = batch

        logging.info(f"clean input shape: {clean_image.shape}")
        logging.info(f"corruption_amount: {corruption_amount}")
        logging.info(f"batch_size = x.shape[0]: {clean_image.shape[0]}")
        logging.info(f"Labels: {label.shape}")

        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=True)
        axs[0].set_title('clean x', fontsize=24)
        axs[1].set_title('noisy x', fontsize=24)
        axs[2].set_title('less noisy x', fontsize=24)

        axs[0].imshow(torchvision.utils.make_grid(clean_image)[0], cmap='Greys')
        axs[1].imshow(torchvision.utils.make_grid(corrupted_image)[0], cmap='Greys')
        axs[2].imshow(torchvision.utils.make_grid(less_corrupted_image)[0], cmap='Greys')
        plt.savefig(os.path.join(save_dir, 'Corruption_pairs_sample.png'), bbox_inches='tight')
        # plt.show()
        # plt.close()

        corruptor = AVAILABLE_CORRUPTORS[config.solver.type](
            config=config,
            transform=config.data.transform)

        # get_initial_corrupted_sample
        n_denoising_steps = config.solver.n_denoising_steps
        initial_corrupted_sample, clean_initial_sample, intermediate_corruption_samples = sampling.get_initial_corrupted_sample(
            trainloader, n_denoising_steps, corruptor)

        utils.save_gif(save_dir, intermediate_corruption_samples, "corruption_init.gif")
        utils.save_video(save_dir, intermediate_corruption_samples, filename="corruption_init.mp4")
        utils.save_png(save_dir, clean_initial_sample, "clean_init.png")

        fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
        axs[0].set_title('clean x', fontsize=24)
        axs[1].set_title('noisy x', fontsize=24)

        axs[0].imshow(torchvision.utils.make_grid(clean_initial_sample)[0], cmap='Greys')
        axs[1].imshow(torchvision.utils.make_grid(initial_corrupted_sample)[0], cmap='Greys')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'Fully_corrupted_sample.png'), bbox_inches='tight')
        # plt.show()
        # plt.close()

if __name__ == '__main__':
    unittest.main()
