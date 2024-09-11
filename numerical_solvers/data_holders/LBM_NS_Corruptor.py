
import numpy as np
import os
import torch
from PIL import Image, ImageFilter


import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import cv2
from abc import ABC, abstractmethod

import taichi as ti
import sys; 
sys.path.insert(0, '../../')

from numerical_solvers.solvers.img_reader import read_img_in_grayscale, normalize_grayscale_image_range
from numerical_solvers.solvers.LBM_NS_Solver import LBM_NS_Solver    
from numerical_solvers.solvers.SpectralTurbulenceGenerator import SpectralTurbulenceGenerator
from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor
    
    
class LBM_NS_Corruptor(BaseCorruptor):
    def __init__(self, initial_dataset, train=True, transform=None, target_transform=None, save_dir='./lbm_mnist'):
        super(LBM_NS_Corruptor, self).__init__(initial_dataset, train, transform, target_transform)

        ti.init(arch=ti.gpu)
        ti_float_precision = ti.f32

        original_pil_image, label = initial_dataset[0]
        channels, nx, ny = self.transform(original_pil_image).shape
        # nx, ny = np_gray_img.shape

        domain_size = (1.0, 1.0)
        grid_size = (nx, ny)
        turb_intensity = 1E-4
        noise_limiter = (-1E-3, 1E-3)
        dt_turb = 5*1E-4 

        # turb_intensity = 1E-3
        # energy_spectrum = lambda k: np.where(np.isinf(k), 0, k)
        
        energy_spectrum = lambda k: np.where(np.isinf(k ** (-5.0 / 3.0)), 0, k ** (-5.0 / 3.0))
        frequency_range = {'k_min': 2.0 * np.pi / min(domain_size), 
                        'k_max': 2.0 * np.pi / (min(domain_size) / 1024)}
        
        spectralTurbulenceGenerator = SpectralTurbulenceGenerator(
            domain_size, grid_size, 
            turb_intensity, noise_limiter,
            energy_spectrum=energy_spectrum, frequency_range=frequency_range, 
            dt_turb=dt_turb, 
            is_div_free = False)
        
        niu = 0.5*1/6
        bulk_visc = 0.5*1/6
        case_name="miau"   
        self.solver = LBM_NS_Solver(
            case_name,
            grid_size,
            niu, bulk_visc,
            spectralTurbulenceGenerator
            )
        
        # solver.init(np_gray_image) 
        
        self.min_lbm_steps = 2 # 1
        self.max_lbm_steps = 50 # 500


        file_path = os.path.join(save_dir, f"{'train' if self.train else 'test'}_data.pt")

        if not os.path.exists(file_path):
            os.makedirs(save_dir, exist_ok=True)
            self._preprocess_and_save_data(initial_dataset, file_path)

    def _corrupt(self, x, lbm_steps):
        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop

        x_noisy = torch.zeros_like(x)  
        np_gray_img = x.numpy()[0,:,:]
        np_gray_img = normalize_grayscale_image_range(np_gray_img, 0.95, 1.05)
        self.solver.init(np_gray_img) 
        self.solver.iterations_counter=0 # reset counter
        self.solver.solve(lbm_steps)
        rho_cpu = self.solver.rho.to_numpy()
        x_noisy[0,:,:] = torch.tensor(rho_cpu) # unsqueeze(0).unsqueeze(0)
        
        return x_noisy    
    
    def _corrupt_pair(self, x, lbm_steps):
        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
        step_difference = 1
                
        np_gray_img = x.numpy()[0,:,:]
        np_gray_img = normalize_grayscale_image_range(np_gray_img, 0.95, 1.05)
        self.solver.init(np_gray_img) 
        self.solver.iterations_counter=0 # reset counter

        self.solver.solve(lbm_steps-step_difference)
        rho_cpu = self.solver.rho.to_numpy()
        x_noisy_t1 = torch.zeros_like(x) 
        x_noisy_t1[0,:,:] = torch.tensor(rho_cpu) # unsqueeze(0).unsqueeze(0)
        
        self.solver.solve(step_difference)
        rho_cpu = self.solver.rho.to_numpy()
        x_noisy_t2 = torch.zeros_like(x) 
        x_noisy_t2[0,:,:] = torch.tensor(rho_cpu) # unsqueeze(0).unsqueeze(0)
        
        return x_noisy_t1, x_noisy_t2    
    
    def _preprocess_and_save_data(self, initial_dataset, file_path):
        data = []
        modified_images = []
        corruption_amounts = []
        labels = []
        
        # for index in range(10000): 
        for index in range(len(initial_dataset)):     # Process all data points
            if index % 100 == 0:
                print(f"Preprocessing (lbm) {index}")
            corruption_amount = np.random.randint(self.min_lbm_steps, 
                                                  self.max_lbm_steps)
            original_pil_image, label = initial_dataset[index]
            original_image = self.transform(original_pil_image)
            modified_image = self._corrupt(original_image, corruption_amount)
            data.append(original_image)
            modified_images.append(modified_image)
            corruption_amounts.append(corruption_amount)
            labels.append(label)
        
        data = torch.stack(data)
        modified_images = torch.stack(modified_images)
        
        corruption_amounts = torch.tensor(corruption_amounts)
        labels = torch.tensor(labels)

        # torch.save((data, targets), save_path)
        torch.save((data, modified_images, corruption_amounts, labels), file_path)
            
