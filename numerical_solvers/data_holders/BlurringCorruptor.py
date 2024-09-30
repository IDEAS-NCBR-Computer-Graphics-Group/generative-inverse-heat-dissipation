import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
import os

from corruptors.BaseCorruptor import BaseCorruptor
from solvers.img_reader import normalize_grayscale_image_range


class BlurringCorruptor(BaseCorruptor):
    def __init__(self, config, transform=None, target_transform=None):
        """
        Initialize the BlurringCorruptor with configuration, transformation, and target transformation.
        
        Args:
            config: Configuration object with relevant settings.
            transform: Optional transform to be applied on a PIL image.
            target_transform: Optional transform to be applied on the target.
        """
        super(BlurringCorruptor, self).__init__(transform, target_transform)        
        # Grayscale normalization range from config
        self.min_init_gray_scale = config.data.min_init_gray_scale
        self.max_init_gray_scale = config.data.max_init_gray_scale
        
        
        self.max_steps = config.solver.max_blurr
        self.min_steps = config.solver.min_blurr
        
    def _corrupt(self, x, corruption_amount, generate_pair=False):
        """
        Corrupts the input image by normalizing and then applying a Gaussian blur using scipy.

        Args:
            x (torch.Tensor): The input image tensor.
            corruption_amount (int): The amount of blur to apply.
            generate_pair (bool): Flag to generate a pair of images (before and after corruption). 
                                  Default is False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Corrupted image or a pair of corrupted images.
        """
        # Convert the input tensor to a numpy array
        np_gray_img = x.numpy()[0, :, :]
        
        # Normalize the grayscale image
        np_gray_img = normalize_grayscale_image_range(np_gray_img, 
                                    self.min_init_gray_scale, self.max_init_gray_scale)

        # take log to spread values 
        # np_gray_img = -np.log10(np_gray_img + 1E-6)  # this does not help 
        
        # Apply Gaussian blur using scipy's gaussian_filter
        blurred_img = gaussian_filter(np_gray_img, sigma=corruption_amount)
        
        # Convert back to Tensor after blurring
        noisy_x = torch.tensor(blurred_img).unsqueeze(0).float()

        if generate_pair:
            # For the pair, use the normalized image before blurring
            less_blurred_img = gaussian_filter(np_gray_img, sigma=(corruption_amount-1.))
            less_noisy_x = torch.tensor(less_blurred_img).unsqueeze(0).float()
            
            return noisy_x, less_noisy_x
        else:
            return noisy_x, None
