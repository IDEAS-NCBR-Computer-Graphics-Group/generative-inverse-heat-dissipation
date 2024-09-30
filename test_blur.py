import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model_code.unet import UNetModel
from model_code import torch_dct
from skimage import data
from torchvision.utils import save_image
import os

# Define DCTBlur
class DCTBlur(nn.Module):
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


class GaussianBlur(nn.Module):
    def __init__(self, blur_sigmas, image_size, device):
        super(GaussianBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        self.kernels = self._create_gaussian_kernels(blur_sigmas, device)
    
    def _create_gaussian_kernels(self, sigmas, device):
        # Determine the maximum kernel size based on the largest sigma
        max_sigma = max(sigmas)
        kernel_size = max(3, int(6 * max_sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        
        # Create a range centered at zero
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        
        kernels = []
        for sigma in sigmas:
            # Create 1D Gaussian for current sigma
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d /= gauss_1d.sum()
            
            # Create 2D Gaussian kernel
            gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
            gauss_2d /= gauss_2d.sum()
            
            kernels.append(gauss_2d)
        
        # Stack all kernels into a single tensor
        kernels = torch.stack(kernels)  # [num_sigmas, K, K]
        kernels = kernels.unsqueeze(1)  # [num_sigmas, 1, K, K]
        return kernels
    
    def forward(self, x, fwd_steps):
        if not torch.is_tensor(fwd_steps):
            fwd_steps = torch.tensor([fwd_steps], device=x.device)
        sigmas = self.blur_sigmas[fwd_steps]
        kernels = self.kernels[fwd_steps]  # [batch, 1, K, K]
        
        # Calculate padding based on kernel size
        padding = (kernels.shape[2] // 2, kernels.shape[3] // 2)
        
        # Apply reflection padding
        # Note: F.pad expects padding in the order (left, right, top, bottom)
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), mode='reflect')
        
        # Apply grouped convolution with zero padding since we've already padded the input
        batch, channels, height, width = x.shape
        x = x.view(1, batch * channels, height, width)
        kernels = kernels.view(batch, 1, kernels.shape[2], kernels.shape[3])
        kernels = kernels.repeat(1, channels, 1, 1)  # [batch*channels, 1, K, K]
        blurred = F.conv2d(x, kernels, groups=batch * channels, padding=0)
        blurred = blurred.view(batch, channels, height - 2 * padding[0], width - 2 * padding[1])
        
        # Since we've subtracted padding in height and width, we need to pad back to original size
        # This is necessary because F.conv2d with padding=0 reduces the spatial dimensions
        # Alternatively, ensure that padding maintains the original size
        # However, since we've used reflection padding, the dimensions should remain the same
        # Therefore, adjust the view accordingly
        blurred = blurred.view(batch, channels, height - 2 * padding[0], width - 2 * padding[1])
        return blurred

   





# Create directory to save images
save_dir = 'blurred_images'
os.makedirs(save_dir, exist_ok=True)

# Load Cameraman image from scikit-image
img = data.camera()
image_size = img.shape[0]
x_single = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

# Initialize DCTBlur for single image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# blur_single = DCTBlur(blur_sigmas=[10.0], image_size=image_size, device=device).to(device)
blur_single = GaussianBlur(blur_sigmas=[10.0], image_size=image_size, device=device).to(device)
x_single = x_single.to(device)

# Apply blur to single image
fwd_steps_single = torch.tensor([0], device=device)  # Single element tensor
blurred_single = blur_single(x_single, fwd_steps=fwd_steps_single)

# Save single blurred image
save_image(blurred_single, os.path.join(save_dir, "blurred_single_sigma10.png"))

# --- Extending to Batch of Images ---

# Define batch size
batch_size = 4

# Create a batch by repeating the single image
x_batch = x_single.repeat(batch_size, 1, 1, 1)  # [batch_size, 1, H, W]

# Define multiple blur sigmas
blur_sigmas_batch = [5.0, 10.0, 15.0, 20.0]

# Initialize DCTBlur with multiple sigmas
blur_batch = GaussianBlur(blur_sigmas=blur_sigmas_batch, image_size=image_size, device=device).to(device)

# Move batch to device
x_batch = x_batch.to(device)

# Create fwd_steps tensor mapping each image to its blur sigma index
fwd_steps_batch = torch.tensor([0, 1, 2, 3], device=device)

# Apply blur to the batch
blurred_batch = blur_batch(x_batch, fwd_steps=fwd_steps_batch)

# Save blurred batch images
for i in range(batch_size):
    sigma = blur_sigmas_batch[i]
    filename = f"blurred_batch_sigma{sigma}.png"
    # save_image(blurred_batch[i], os.path.join(save_dir, filename))
    # Convert tensors to images
    # input_img = blurred_batch[i].squeeze().cpu().numpy().astype(np.uint8)
    blurred_img = blurred_batch[i].squeeze().cpu().detach().numpy()
    blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)
    Image.fromarray(blurred_img).save(f'blurred_gauss_{i}.png')

print(f"Blurred images saved to '{save_dir}' directory.")