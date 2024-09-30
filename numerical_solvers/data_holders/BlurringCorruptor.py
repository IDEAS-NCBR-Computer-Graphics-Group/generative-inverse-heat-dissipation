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
        
    def _corrupt(self, x, fwd_steps, generate_pair=False):
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
        # TODO: this is adopted to hopefully match the IHD config
        # TODO: read it from config
        K = 50
        blur_sigma_max = 20
        blur_sigma_min = 0.5
        blur_schedule = np.exp(np.linspace(
            np.log(blur_sigma_min), np.log(blur_sigma_max), K))
        
        blur_schedule = np.array([0] + list(blur_schedule))

        sigmas = blur_schedule[fwd_steps]  
        corruption_amount = np.sqrt(sigmas**2)
        
        less_sigmas = blur_schedule[fwd_steps-1] 
        less_corruption_amount = np.sqrt(less_sigmas**2)
        blurred_img = gaussian_filter(np_gray_img, sigma=corruption_amount)
        
        # Convert back to Tensor after blurring
        noisy_x = torch.tensor(blurred_img).unsqueeze(0).float()

        if generate_pair:
            # For the pair, use the normalized image before blurring
            # step_size = 1
            # less_blurred_img = gaussian_filter(np_gray_img, sigma=(corruption_amount-self.step_size))
            
            less_blurred_img = gaussian_filter(np_gray_img, sigma=less_corruption_amount)
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
        file_name = f"{'train' if is_train_dataset else 'test'}_data.pt"
        file_path = os.path.join(save_dir, file_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        if os.path.exists(file_path):
            warnings.warn(f"[EXIT] Data not generated. Reason: file exists {file_path}")
            return
        
        data = []
        modified_images = [] 
        corruption_amounts = []
        labels = []

        # Only needed if processing pairs
        pre_modified_images = [] if process_pairs else None  

        dataset_length = len(initial_dataset)
        if not process_all:
            dataset_length = 500  # Process just a bit 
        
        for index in range(dataset_length):
            if index % 100 == 0:
                print(f"Preprocessing (blurring) {index}")
            
            # corruption_amount = np.random.randint(self.min_blurr, self.max_blurr) # ints
            
            corruption_amount = np.random.uniform(low=self.min_blurr, high=self.max_blurr, size=None)
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
        data = torch.stack(data)
        modified_images = torch.stack(modified_images)
        corruption_amounts = torch.tensor(corruption_amounts)
        labels = torch.tensor(labels)

        if process_pairs:
            pre_modified_images = torch.stack(pre_modified_images)
            torch.save((data, modified_images, pre_modified_images, corruption_amounts, labels), file_path)
        else:
            torch.save((data, modified_images, corruption_amounts, labels), file_path)
