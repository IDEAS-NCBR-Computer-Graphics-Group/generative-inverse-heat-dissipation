import numpy as np
import os
import torch
from abc import ABC
import warnings

import taichi as ti
from numerical_solvers.solvers.img_reader import normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver    
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor

class LBM_NS_Corruptor(BaseCorruptor):
    def __init__(self, grid_size, train=True, transform=None, target_transform=None):
        super(LBM_NS_Corruptor, self).__init__(train, transform, target_transform)

        ti.init(arch=ti.gpu)
        domain_size = (1.0, 1.0)

        turb_intensity = 1E-4
        noise_limiter = (-1E-3, 1E-3)
        dt_turb = 5 * 1E-4 
        energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
        frequency_range = {
            'k_min': 2.0 * np.pi / min(domain_size), 
            'k_max': 2.0 * np.pi / (min(domain_size) / 1024)
        }

        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            domain_size, grid_size, 
            turb_intensity, noise_limiter,
            energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
            dt_turb=dt_turb, 
            is_div_free=False
        )
        
        niu = 0.5 * 1/6
        bulk_visc = 0.5 * 1/6
        case_name = "miau"   
        self.solver = LBM_NS_Solver(
            case_name,
            grid_size,
            niu, bulk_visc,
            spectralTurbulenceGenerator
        )

        self.min_lbm_steps = 2 
        self.max_lbm_steps = 50
        
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
        np_gray_img = normalize_grayscale_image_range(np_gray_img, 0.95, 1.05)
        
        self.solver.init(np_gray_img)
        self.solver.iterations_counter = 0  # Reset counter

        # Solve up to lbm_steps - step_difference if generating pairs, else directly to lbm_steps
        self.solver.solve(lbm_steps - step_difference if generate_pair else lbm_steps)
        rho_cpu = self.solver.rho.to_numpy()
        x_noisy_pre_t = torch.tensor(rho_cpu).unsqueeze(0)

        if generate_pair:
            # Solve the remaining steps for the pair generation
            self.solver.solve(step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            x_noisy_t = torch.tensor(rho_cpu).unsqueeze(0)
            return x_noisy_pre_t, x_noisy_t
        
        return x_noisy_pre_t, None

    def _preprocess_and_save_data(self, initial_dataset, save_dir, process_pairs=False, process_all=True):
        """
        Preprocesses data and saves it to the specified directory.

        Args:
            initial_dataset (list): The initial dataset containing images and labels.
            save_dir (str): The directory to save the preprocessed data.
            process_pairs (bool): Flag indicating whether to process pairs of images (True) 
                                  or single corrupted images (False). Default is False.
        """
        file_name = f"{'train' if self.train else 'test'}_data.pt"
        file_path = os.path.join(save_dir, file_name)
 
 
        if os.path.exists(file_path):
            warnings.warn("[EXIT] Data not generated. Reason: file exist {file_path} ")
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
                print(f"Preprocessing (lbm) {index}")
            
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

        # Convert lists to tensors ### TODO: tensors dont match the order of transforms in the original ihd code
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