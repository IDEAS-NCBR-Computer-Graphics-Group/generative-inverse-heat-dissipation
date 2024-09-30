import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
import os

from zmq import device
from models import torch_dct

from corruptors.BaseCorruptor import BaseCorruptor
from solvers.img_reader import normalize_grayscale_image_range

# Define DCTBlur
class DCTBlur(torch.nn.Module):
    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi * torch.linspace(0, image_size-1, image_size).to(device) / image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        sigmas = self.blur_sigmas[fwd_steps]
        if x.dim() == 4:
            sigmas = sigmas[:, None, None, None]
        elif x.dim() == 3:
            sigmas = sigmas[:, None, None]
        t = sigmas**2 / 2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs *= torch.exp(-self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')


class BlurringCorruptorIHD(BaseCorruptor):
    def __init__(self, config, transform=None, target_transform=None, device=torch.device('cuda')):
        """
        Initialize the BlurringCorruptor with configuration, transformation, and target transformation.
        
        Args:
            config: Configuration object with relevant settings.
            transform: Optional transform to be applied on a PIL image.
            target_transform: Optional transform to be applied on the target.
        """
        super(BlurringCorruptorIHD, self).__init__(transform, target_transform)   
        self.device = device    
        self.config = config 
        # Grayscale normalization range from config
        self.min_init_gray_scale = config.blur.data.min_init_gray_scale
        self.max_init_gray_scale = config.blur.data.max_init_gray_scale
        
        self.max_steps = config.blur.solver.max_blurr
        self.min_steps = config.blur.solver.min_blurr
        scales = config.model.blur_schedule

        self.blur_batch = DCTBlur(blur_sigmas=scales, image_size=config.data.image_size, device=device).to(device)
        
    
    def get_label_sampling_function(self, K):
        return lambda batch_size, device: torch.randint(1, K, (batch_size,), device=device)

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
        blurred_batch = self.blur_batch(x, fwd_steps).float()
        less_blurred_batch = self.blur_batch(x, fwd_steps-1).float()
        
        if generate_pair:
            return blurred_batch, less_blurred_batch
        else:
            return blurred_batch, None

    def _preprocess_and_save_data(self, dataloader, save_dir, is_train_dataset: bool, process_pairs=False):
        """
        Preprocesses data and saves it to the specified directory.

        Args:
            initial_dataset (list): The initial dataset containing images and labels.
            save_dir (str): The directory to save the preprocessed data.
            process_pairs (bool): Flag indicating whether to process pairs of images (True) 
                                  or single corrupted images (False). Default is False.
        """
        iter_data = iter(dataloader)
        batch_size = self.config.training.batch_size

        split = 'train' if is_train_dataset else 'test'

        split_save_dir = os.path.join(save_dir, split)
        if os.path.exists(split_save_dir):
            warnings.warn(f"[EXIT] Data not generated. Reason: file exist {save_dir} and is not empty.")
            return
        os.makedirs(split_save_dir)



        for i in tqdm(range(0, len(dataloader) * batch_size, batch_size)):
            batch = next(iter_data)

            per_sample_corrupted_images = [] 
            per_sample_corruption_amounts = []
            labels = []
            pre_corrupted_images = [] if process_pairs else None     

            image, label = batch
            image = self.transform(image)
            label_sampling_fn = self.get_label_sampling_function(self.config.model.K)

            fwd_steps = label_sampling_fn(image.shape[0], self.device)

            if not process_pairs:
                per_sample_corrupted_images.append(image)
                per_sample_corruption_amounts.append(torch.zeros_like(fwd_steps).cpu())
                labels.append(label)

            for k in range(1, int(self.max_steps)):

                label_sampling_fn = self.get_label_sampling_function(self.config.model.K)

                corrupted_image, pre_corrupted_image = self._corrupt(
                    image.to(self.device),
                    fwd_steps,
                    generate_pair=process_pairs
                    )
                image = corrupted_image

                per_sample_corrupted_images.append(corrupted_image.cpu())
                per_sample_corruption_amounts.append(fwd_steps.cpu())
                labels.append(label)

                if process_pairs:
                    pre_corrupted_images.append(pre_corrupted_image)

            for bi in range(len(per_sample_corrupted_images[0])):
                file_path = os.path.join(split_save_dir, f'data_point_{i+bi}.pt')

                data_b = []
                ca_b = []
                l_b = []
                pci_b = [] if process_pairs else None     


                for k in range(len(per_sample_corrupted_images)):
                
                    data_b.append(per_sample_corrupted_images[k][bi])
                    ca_b.append(per_sample_corruption_amounts[k][bi])
                    l_b.append(labels[k][bi])

                    if process_pairs:
                        pci_b.append(pre_corrupted_image[k][bi])


                data = torch.stack(data_b)
                ca = torch.stack(ca_b)
                l = torch.stack(l_b)


                if process_pairs:
                    pci = torch.stack(pci_b)
                    torch.save((data, pci, ca, l), file_path)
                else:
                    torch.save((data, ca, l), file_path)