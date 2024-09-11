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
    def __init__(self, train=True, transform=None, target_transform=None, download=False, load_dir='./data/corrupted_MNIST/lbm_ns'):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        
        file_path = os.path.join(load_dir, f"{'train' if self.train else 'test'}_data.pt")
        
        if os.path.exists(file_path):
            self.data, self.targets = self._load_data(file_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)    

    def _load_data(self, file_path):    
        loaded_data = torch.load(file_path)
        if len(loaded_data) == 4:
            # Original mode with no pre-modified images
            data, modified_images, corruption_amounts, labels = loaded_data
            targets = list(zip(modified_images, corruption_amounts, labels))
        elif len(loaded_data) == 5:
            # Pair mode with pre-modified images
            data, modified_images, pre_modified_images, corruption_amounts, labels = loaded_data
            targets = list(zip(modified_images, pre_modified_images, corruption_amounts, labels))
        else:
            raise ValueError(f"Unexpected data format in {file_path}, expected 4 or 5 elements, got {len(loaded_data)}")

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        original_image = self.data[index]
        
        if len(self.targets[index]) == 4:  # Check if there are pre-modified images
            modified_image, pre_modified_image, corruption_amount, label = self.targets[index]
            
            # Apply the transformations if any
            if self.transform is not None:
                original_image = self.transform(original_image)
                modified_image = self.transform(modified_image)
                pre_modified_image = self.transform(pre_modified_image)

            return original_image, (modified_image, pre_modified_image, corruption_amount.item(), label.item())
        else:
            modified_image, corruption_amount, label = self.targets[index]
            
            # Apply the transformations if any
            if self.transform is not None:
                original_image = self.transform(original_image)
                modified_image = self.transform(modified_image)

            return original_image, (modified_image, corruption_amount.item(), label.item())
