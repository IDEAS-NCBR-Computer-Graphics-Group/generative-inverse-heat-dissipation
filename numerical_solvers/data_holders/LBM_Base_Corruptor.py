import numpy as np
import os
import torch
from abc import ABC
import warnings
import logging

import taichi as ti
from numerical_solvers.solvers.img_reader import normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_SolverBase import LBM_SolverBase
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor

from scripts.utils import load_config_from_path, setup_logging


class LBM_Base_Corruptor(BaseCorruptor):
    def __init__(self, config, transform=None, target_transform=None):
        super(LBM_Base_Corruptor, self).__init__(transform, target_transform)
        ti.init(arch=ti.gpu, random_seed=config.seed)

        # Set LBM steps (can be made configurable too)
        self.min_steps = config.solver.min_fwd_steps
        self.max_steps = config.solver.max_fwd_steps

        self.corrupt_sched = config.solver.corrupt_sched

        self.min_init_gray_scale = config.solver.min_init_gray_scale
        self.max_init_gray_scale = config.solver.max_init_gray_scale

        self._intermediate_samples = None


    @property
    def intermediate_samples(self):
        return self._intermediate_samples

    def _corrupt(self, x, steps, generate_pair=False):
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
        np_gray_img = x.numpy()[0, :, :]
        np_gray_img = normalize_grayscale_image_range( # rescale to fit solver stability range
            np_gray_img, self.min_init_gray_scale, self.max_init_gray_scale)
    
        self.solver.init(np_gray_img)
        self.solver.iterations_counter = 0  # Reset counter

        self._intermediate_samples = torch.empty((steps + 1, *x.shape))
        self._intermediate_samples[0] = torch.tensor( # rescale for preview
            normalize_grayscale_image_range(np_gray_img, 0., 1.)).unsqueeze(0).clone()

        # s = 0
        # print(self.corrupt_sched)
        for i in range(steps-1):
            step_difference = self.corrupt_sched[i] - self.corrupt_sched[i-1]
            if i == 0: step_difference = self.corrupt_sched[0]
            self.solver.solve(step_difference)
            rho_cpu = self.solver.rho.to_numpy()
            rho_cpu = normalize_grayscale_image_range(rho_cpu, 0., 1.)
            self._intermediate_samples[i+1] = torch.tensor(rho_cpu).unsqueeze(0)
            # print(i)
            # print(f'sd: {step_difference}')
            # s += step_difference
        # print(steps)

        step_difference = self.corrupt_sched[steps] - self.corrupt_sched[steps-1]
        # print(f'sd: {step_difference}')
        self.solver.solve(step_difference)
        rho_cpu = self.solver.rho.to_numpy()
        rho_cpu = normalize_grayscale_image_range(rho_cpu, 0., 1.)
        self._intermediate_samples[steps] = torch.tensor(rho_cpu).unsqueeze(0)

        logging.info(f"Corruptor.solver run for iterations: {self.solver.iterations_counter}")
        # s += step_difference
        # print(f'solver steps:{s}')

        if generate_pair:
            noisy_x = self._intermediate_samples[-1].clone()
            less_noisy_x = self._intermediate_samples[-2].clone()
            return noisy_x, less_noisy_x
        else:
            noisy_x = self._intermediate_samples[-1].clone()
            return noisy_x, None

    def _preprocess_and_save_data(self,
                                  initial_dataset,
                                  save_dir,
                                  is_train_dataset: bool,
                                  process_pairs=False, process_all=True,  process_images=False):
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
            logging.warning(f"[EXIT] Data not generated. Reason: file exist {file_path}")
            return

        data = []
        modified_images = [] 
        corruption_amounts = []
        labels = []

        # Only needed if processing pairs
        pre_modified_images = [] if process_pairs else None  

        dataset_length = len(initial_dataset)
        if not process_all:
            dataset_length = 256 # process just a bit 
            logging.info(f"Preprocessing (lbm) a piece of dataset: {dataset_length}")
            
        for index in range(dataset_length):
            if index % 100 == 0:
                logging.info(f"Preprocessing (lbm) {index}")
            
            # max_steps is excluded from tossing
            corruption_amount = np.random.randint(self.min_steps, self.max_steps)
            original_pil_image, label = initial_dataset[index]
            if process_images:
                original_pil_image = np.transpose(original_pil_image, [1, 2, 0])
            original_image = self.transform(original_pil_image)

            # Use the unified corrupt function and ignore the second value if not needed
            modified_image, less_modified_image = self._corrupt(original_image, corruption_amount, generate_pair=process_pairs)
            self.solver.turbulenceGenerator.randomize()
            data.append(original_image)
            modified_images.append(modified_image)
            corruption_amounts.append(corruption_amount)
            if not process_images:
                labels.append(label)

            if process_pairs:
                pre_modified_images.append(less_modified_image)

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
            torch.save((data, modified_images, pre_modified_images, corruption_amounts, labels if not process_images else None), file_path)
        else:
            torch.save((data, modified_images, corruption_amounts, labels if not process_images else None), file_path)

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