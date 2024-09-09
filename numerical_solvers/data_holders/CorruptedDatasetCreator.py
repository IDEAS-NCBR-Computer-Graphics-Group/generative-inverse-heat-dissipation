
# %% dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
import os, errno
import torchvision
import numpy as np
from matplotlib import pyplot as plt

import matplotlib
from timeit import default_timer as timer


from BlurringCorruptor import BlurringCorruptor
# from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from LBM_NS_Corruptor import LBM_NS_Corruptor
from CorruptedDataset import CorruptedDataset

import sys; 
from pathlib import Path
# sys.path.insert(0, '../../')
                                
# %% dataset
# Create the DiffusedMNIST dataset
if __name__ == '__main__': 
    

    print(f"Current working directory \t {os.getcwd()}")
    
    current_file_path = Path(__file__).resolve()
    
    # Determine the base folder (project root)
    base_folder = current_file_path.parents[2]  # Adjust the number depending on your project structure

    print(f"Base folder: {base_folder}")
    

    input_data_dir = os.path.join(base_folder, "data")
    output_data_dir = os.path.join(input_data_dir, 'corrupted_MNIST')
    
    # https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    original_dataloader = DataLoader(torchvision.datasets.MNIST(root=input_data_dir, 
                                    train=True, download=True, transform=transform), 
                                    batch_size=8, shuffle=True)

    x, y = next(iter(original_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', y.shape)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    
    # Define the transformations
    corrupted_dataset_dir = os.path.join(output_data_dir, 'blurred')
    blurringCorruptor = BlurringCorruptor(initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True), 
                                          train=True, 
                                          transform=transform, 
                                          save_dir=corrupted_dataset_dir)
    diffused_mnist_train = CorruptedDataset(load_dir=corrupted_dataset_dir, transform=None, target_transform=None)

    corrupted_dataloader = DataLoader(diffused_mnist_train, batch_size=8, shuffle=True)
    x, (y, corruption_amount, label) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', y[1].shape)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    plt.imshow(torchvision.utils.make_grid(y)[0], cmap='Greys');

    # %% lbmize
    corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns')
    start = timer()
    lbmCorruptor = LBM_NS_Corruptor(initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True), 
                                train=True, 
                                transform=transform, 
                                save_dir=corrupted_dataset_dir)
    end = timer()
    print(f"Time in seconds: {end - start:.2f}")

    lbm_mnist_train = CorruptedDataset(load_dir=corrupted_dataset_dir, transform=None, target_transform=None)

    corrupted_dataloader = DataLoader(lbm_mnist_train, batch_size=8, shuffle=True)
    x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', labels)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    # plt.imshow(torchvision.utils.make_grid(y)[0], cmap='Greys');
    
    plt.imshow(torchvision.utils.make_grid(y)[0], 
               norm=matplotlib.colors.Normalize(vmin=0.95, vmax=1.05),
               cmap='Greys');
    


# %%
