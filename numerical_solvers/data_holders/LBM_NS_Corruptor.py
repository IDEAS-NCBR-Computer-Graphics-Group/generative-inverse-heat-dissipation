import numpy as np
import os
import torch
import warnings

import taichi as ti
from numerical_solvers.solvers.img_reader import normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver    
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor
from configs.mnist.lbm_ns_turb_config import LBMConfig
import logging

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

    def _preprocess_and_save_data(self, initial_dataset, save_dir, is_train_dataset: bool, process_pairs=False, process_all=True):
        """
        Preprocesses data and saves it to the specified directory.

        Args:
            initial_dataset (list): The initial dataset containing images and labels.
            save_dir (str): The directory to save the preprocessed data.
            process_pairs (bool): Flag indicating whether to process pairs of images (True) 
                                  or single corrupted images (False). Default is False.
        """
        file_name = f"{'train' if is_train_dataset else 'test'}_data.pt"
        file_path = os.path.join(save_dir, file_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if os.path.exists(file_path):
            warnings.warn(f"[EXIT] Data not generated. Reason: file exist {file_path} ")
            return
      

        data = []
        modified_images = [] 
        corruption_amounts = []
        labels = []

        # Only needed if processing pairs
        pre_modified_images = [] if process_pairs else None  

        dataset_length = len(initial_dataset)
        if not process_all:
            dataset_length = 500 # process just a bit 
            
        for index in range(dataset_length):
            if index % 100 == 0:
                logging.info(f"[LBM] Preprocessing: {index}.")
            
            corruption_amount = np.random.randint(self.min_lbm_steps, self.max_lbm_steps)
            original_pil_image, label = initial_dataset[index]
            original_image = self.transform(original_pil_image)

            # Use the unified corrupt function and ignore the second value if not needed
            modified_image, pre_modified_image = self._corrupt(original_image, corruption_amount, generate_pair=process_pairs)

            data.append(original_image)
            modified_images.append(modified_image)
            corruption_amounts.append(corruption_amount)
            labels.append(label)

            if process_pairs:
                pre_modified_images.append(pre_modified_image)

        # Convert lists to tensors 
        # Be carfull, tensors dont match the order of transforms in the original ihd code
        # You are likely to use the following later 
            #  transform = [
            #     transforms.ToPILImage(), 
            #     transforms.Resize(config.data.image_size),
            #     transforms.CenterCrop(config.data.image_size),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor()
            #     ]
            
        data = torch.stack(data)
        modified_images = torch.stack(modified_images)
        corruption_amounts = torch.tensor(corruption_amounts)
        labels = torch.tensor(labels)

        if process_pairs:
            pre_modified_images = torch.stack(pre_modified_images)
            torch.save((data, modified_images, pre_modified_images, corruption_amounts, labels), file_path)
        else:
            torch.save((data, modified_images, corruption_amounts, labels), file_path)

        # Convert lists to ndarrays
        # data = np.array(data)
        # modified_images = np.array(modified_images)
        # corruption_amounts = np.array(corruption_amounts)
        # labels = np.array(labels)

        # print(f"Writing to {file_path}")
        # if process_pairs:
        #     pre_modified_images = np.array(pre_modified_images)
        #     torch.save((data, modified_images, pre_modified_images, corruption_amounts, labels), file_path)
        # else:
        #     torch.save((data, modified_images, corruption_amounts, labels), file_path)