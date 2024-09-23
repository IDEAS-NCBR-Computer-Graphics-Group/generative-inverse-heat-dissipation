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
        
        
        self.min_blurr = config.solver.min_blurr
        self.max_blurr = config.solver.max_blurr
        
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

    def _preprocess_and_save_data(self, initial_dataset, save_dir, is_train_dataset: bool, process_pairs=False, process_all=True):
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

            # corruption_amount = np.random.randint(self.min_blurr, self.max_blurr) # ints
            corruption_amount = np.random.uniform(low=self.min_blurr, high=self.max_blurr, size=None)
            image, label = initial_dataset[i]
            original_image = self.transform(image)

            # Use the unified corrupt function and ignore the second value if not needed
            corrupted_image, pre_corrupted_image = self._corrupt(
                original_image,
                corruption_amount,
                generate_pair=process_pairs
                )

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
