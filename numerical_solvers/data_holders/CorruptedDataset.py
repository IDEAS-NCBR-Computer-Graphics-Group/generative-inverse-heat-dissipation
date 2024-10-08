# %% dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
import os, errno
import torchvision
import numpy as np
from matplotlib import pyplot as plt


# %% dataset

class CorruptedDataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, download=False, load_dir=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.handle_labels = True
        
        file_path = os.path.join(load_dir, f"{'train' if self.train else 'test'}_data.pt")
        
        if os.path.exists(file_path):
            self.data, self.targets = self._load_data(file_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)    

    def _load_data(self, file_path):    
        loaded_data = torch.load(file_path, weights_only=True)
        if len(loaded_data) == 4:
            # Original mode with no pre-modified images
            data, corrupted_images, corruption_amounts, labels = loaded_data
            if labels is not None:
                targets = list(zip(corrupted_images, corruption_amounts, labels))
            else:
                self.handle_labels = False
                targets = list(zip(corrupted_images, corruption_amounts, torch.zeros_like(corruption_amounts)))
        elif len(loaded_data) == 5:
            # Pair mode with pre-modified images
            data, corrupted_images, less_corrupted_images, corruption_amounts, labels = loaded_data
            if labels is not None:
                targets = list(zip(corrupted_images, less_corrupted_images, corruption_amounts, labels))
            else:
                self.handle_labels = False
                targets = list(zip(corrupted_images, less_corrupted_images, corruption_amounts, torch.zeros_like(corruption_amounts)))
        else:
            raise ValueError(f"Unexpected data format in {file_path}, expected 4 or 5 elements, got {len(loaded_data)}")

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        original_image = self.data[index]
        
        if len(self.targets[index]) == 4:  # Check if there are pre-modified images
            if self.handle_labels:
                corrupted_image, less_corrupted_image, corruption_amount, label = self.targets[index]
            else:
                corrupted_image, less_corrupted_image, corruption_amount, label = self.targets[index]

            # Convert numpy array to PIL Image directly with mode 'L' for grayscale images
            # original_image = Image.fromarray(original_image.astype('uint8'), mode='RGB')
            # modified_image = Image.fromarray(modified_image.astype('uint8'), mode='RGB')
            # pre_modified_image = Image.fromarray(pre_modified_image.astype('uint8'), mode='RGB')

            # Apply the transformations if any
            if self.transform is not None:
                original_image = self.transform(original_image)
                corrupted_image = self.transform(corrupted_image)
                less_corrupted_image = self.transform(less_corrupted_image)

            if self.handle_labels:
                return original_image, (corrupted_image, less_corrupted_image, corruption_amount.item(), label.item())
            else:
                return original_image, (corrupted_image, less_corrupted_image, corruption_amount.item(), label.item())
        
        else:
            corrupted_image, corruption_amount, label = self.targets[index]
            
            # Apply the transformations if any
            if self.transform is not None:
                original_image = self.transform(original_image)
                corrupted_image = self.transform(corrupted_image)

            if self.handle_labels:
                return original_image, (corrupted_image, corruption_amount.item(), label.item())
            else:
                return original_image, (corrupted_image, corruption_amount.item(), label.item())
