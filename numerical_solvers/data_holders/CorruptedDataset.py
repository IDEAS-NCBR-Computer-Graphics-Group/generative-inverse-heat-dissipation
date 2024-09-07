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
    def __init__(self, train=True, transform=None, target_transform=None, download=False, load_dir='./data/corrupted_data/lbm_mnist'):
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
        data, modified_images, corruption_amounts, labels = torch.load(file_path)
        targets = list(zip(modified_images, corruption_amounts, labels))
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        original_image = self.data[index]
        modified_image, corruption_amount, label = self.targets[index]

        # Apply the transformations if any
        # https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
        if self.transform is not None:
            original_image = self.transform(original_image)
            modified_image = self.transform(modified_image)

        return original_image, (modified_image, corruption_amount.item(), label.item())
