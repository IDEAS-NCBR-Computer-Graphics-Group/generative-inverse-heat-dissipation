import taichi as ti
import numpy as np
import torch
from tqdm import tqdm
import warnings
import os

from solvers.img_reader import normalize_grayscale_image_range
from solvers.LBM_NS_Solver import LBM_NS_Solver    
from solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from corruptors.BaseCorruptor import BaseCorruptor
from configs.mnist.lbm_ns_turb_config import LBMConfig

class LBM_NS_Corruptor(BaseCorruptor):
    def __init__(self, config: LBMConfig, transform=None, target_transform=None):
        super(LBM_NS_Corruptor, self).__init__(transform, target_transform)

        ti.init(arch=ti.gpu)

        grid_size = (config.data.image_size, config.data.image_size)
        
        # energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
        # frequency_range = {'k_min': config.solver.k_min, 'k_max': config.solver.k_max }

        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            config.solver.domain_size, grid_size, 
            config.solver.turb_intensity, config.solver.noise_limiter,
            energy_spectrum=config.solver.energy_spectrum, 
            frequency_range={'k_min': config.solver.k_min, 'k_max': config.solver.k_max}, 
            dt_turb=config.solver.dt_turb, 
            is_div_free=False
        )
        
        # LBM NS Solver setup        
        # Instantiate the LBM NS Solver using the config and spectral turbulence generator
        self.solver = LBM_NS_Solver(
            "miau",
            grid_size,
            config.solver.niu, config.solver.bulk_visc,
            spectralTurbulenceGenerator
        )

        # Set LBM steps (can be made configurable too)
        self.min_lbm_steps = config.solver.min_lbm_steps
        self.max_lbm_steps = config.solver.max_lbm_steps
        
        self.min_init_gray_scale = config.data.min_init_gray_scale
        self.max_init_gray_scale = config.data.max_init_gray_scale
        
        
    def _corrupt(self, x, lbm_steps, generate_pair=False):
        """
        Corrupts the input image using LBM solver.

        Args:
            x (torch.Tensor): The input image tensor.
            lbm_steps (int): The number of steps for LBM solver to run.
            generate_pair (bool): Flag to generate a pair of images (before and after a step difference). 
                                  Default is False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Corrupted image or a pair of corrupted images.
        """
        step_difference = 1  # Step difference for generating pairs
        np_gray_img = x.numpy()[0, :, :]
        np_gray_img = normalize_grayscale_image_range(
            np_gray_img, self.min_init_gray_scale, self.max_init_gray_scale)
        
        self.solver.init(np_gray_img)
        self.solver.iterations_counter = 0  # Reset counter

        if generate_pair:
            # Solve up to (lbm_steps - step_difference) if generating pairs
            self.solver.solve(lbm_steps - step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            less_noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            
            # Solve the remaining steps for the pair generation
            self.solver.solve(step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            return noisy_x, less_noisy_x,
        else:
            # Solve up to lbm_steps - step_difference if generating pairs, else directly to lbm_steps
            self.solver.solve(lbm_steps)
            rho_cpu = self.solver.rho.to_numpy()
            noisy_x = torch.tensor(rho_cpu).unsqueeze(0)
            return noisy_x, None

    def _preprocess_and_save_data(self, initial_dataset, save_dir, is_train_dataset: bool, process_pairs=False):
        """
        Preprocesses data and saves it to the specified directory.

        Args:
            initial_dataset (list): The initial dataset containing images and labels.
            save_dir (str): The directory to save the preprocessed data.
            process_pairs (bool): Flag indicating whether to process pairs of images (True) 
                                  or single corrupted images (False). Default is False.
        """
        split = 'train' if is_train_dataset else 'test'

        split_save_dir = os.path.join(save_dir, split)
        if os.path.exists(split_save_dir):
            warnings.warn(f"[EXIT] Data not generated. Reason: file exist {save_dir} and is not empty.")
            return
        os.makedirs(split_save_dir)
            
        for i in tqdm(range(len(initial_dataset))):
            file_path = os.path.join(split_save_dir, f'data_point_{i}.pt')

            corruption_amount = np.random.randint(self.min_lbm_steps, self.max_lbm_steps)
            image, label = initial_dataset[i]
            image = self.transform(image)

            corrupted_image, pre_corrupted_image = self._corrupt(
                image,
                corruption_amount,
                generate_pair=process_pairs
                )
            corruption_amount = torch.tensor(corruption_amount)

            if process_pairs:
                torch.save(
                    (
                    image,
                    corrupted_image,
                    pre_corrupted_image,
                    corruption_amount,
                    label
                    ),
                    file_path
                    )
            else:
                torch.save(
                    (
                    image,
                    corrupted_image,
                    corruption_amount,
                    label
                    ),
                    file_path
                    )