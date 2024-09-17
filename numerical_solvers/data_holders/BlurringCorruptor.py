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


import taichi as ti
import sys; 
sys.path.insert(0, '..')

# from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor
from numerical_solvers.data_holders.BaseCorruptor import BaseCorruptor

class BlurringCorruptor(BaseCorruptor):
    def __init__(self, initial_dataset, train=True, transform=None, target_transform=None, save_dir='./blurred_mnist'):
        super(BlurringCorruptor, self).__init__(train, transform, target_transform)
        
        file_path = os.path.join(save_dir, f"{'train' if self.train else 'test'}_data.pt")
        if not os.path.exists(file_path):
            os.makedirs(save_dir, exist_ok=True)
            self._preprocess_and_save_data(initial_dataset, file_path)
                  
    def _corrupt(self, pil_image, corruption_amount):
        # pil_image = transforms.ToPILImage()(image)
        modified_pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=corruption_amount))
        return modified_pil_image

    def _preprocess_and_save_data(self, initial_dataset, file_path):
        data = []
        modified_images = []
        corruption_amounts = []
        labels = []
        
        # for index in range(10000): 
        for index in range(len(initial_dataset)):     # Process all data points
            if index % 1000 == 0:
                print(f"Preprocessing (blurring) {index}")
            corruption_amount = np.random.randint(2,7)
            original_pil_image, label = initial_dataset[index]
            modified_image = self._corrupt(original_pil_image, corruption_amount)
            
            data.append(self.transform(original_pil_image))
            modified_images.append(self.transform(modified_image))
            corruption_amounts.append(corruption_amount)
            labels.append(label)
        
        data = torch.stack(data)
        modified_images = torch.stack(modified_images)
        
        corruption_amounts = torch.tensor(corruption_amounts)
        labels = torch.tensor(labels)

        # torch.save((data, targets), save_path)
        torch.save((data, modified_images, corruption_amounts, labels), file_path)
        