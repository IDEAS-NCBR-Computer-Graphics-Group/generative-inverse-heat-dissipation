
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

import sys
from pathlib import Path

from numerical_solvers.data_holders.BlurringCorruptor import BlurringCorruptor
# from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets

# sys.path.insert(0, '../../')
                                
# %% dataset
# Create the DiffusedMNIST dataset
if __name__ == '__main__': 
    print(f"Current working directory \t {os.getcwd()}")
    current_file_path = Path(__file__).resolve()
    # Determine the base folder (project root)
    base_folder = current_file_path.parents[2]  # Adjust the number depending on your project structure
    print(f"Base folder: {base_folder}")
    
    is_train_dataset = False

    input_data_dir = os.path.join(base_folder, "data")
    output_data_dir = os.path.join(input_data_dir, 'corrupted_MNIST')
    
    # https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    original_dataloader = DataLoader(torchvision.datasets.MNIST(root=input_data_dir, 
                                    train=is_train_dataset, download=True, transform=transform), 
                                    batch_size=8, shuffle=True)

    x, y = next(iter(original_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', y.shape)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    
   

    # %% lbmize
    
    start = timer()
    initial_dataset = datasets.MNIST(root=input_data_dir, train=is_train_dataset, download=True)
    
    lbm_ns_Corruptor = LBM_NS_Corruptor(
        grid_size = initial_dataset[0][0].size,
        train=is_train_dataset, 
        transform=transform)
    
    corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns_pair')
    process_pairs=True
    lbm_ns_Corruptor._preprocess_and_save_data(
        initial_dataset=initial_dataset,
        save_dir=corrupted_dataset_dir,
        process_pairs = process_pairs,
        process_all=False)
    
    end = timer()
    print(f"Time in seconds: {end - start:.2f}")

    
    # use same transform as in ihd code
    
    # transform = [
    #             torchvision.transforms.ToPILImage()
    #             transforms.Resize(28),
    #             transforms.CenterCrop(28),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor()
    #             ]
    # transform = transforms.Compose(transform)
    
    transform = None # the dataset is saved as torchtensor
    lbm_mnist_pairs = CorruptedDataset(train=is_train_dataset, 
                                       transform=transform, 
                                       target_transform=None, 
                                       load_dir=corrupted_dataset_dir)



    
    corrupted_dataloader = DataLoader(lbm_mnist_pairs, batch_size=8, shuffle=True)
    if process_pairs:
        print(f"==processing pairs===")
        x, (y, pre_y, corruption_amount, labels) = next(iter(corrupted_dataloader))
        # alternatively
        x, batch = ihd_datasets.prepare_batch(iter(corrupted_dataloader),'cpu')
        y, pre_y, corruption_amount, labels = batch
    else:
        x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', labels)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    plt.imshow(torchvision.utils.make_grid(y)[0], cmap='Greys');
    
    plt.imshow(torchvision.utils.make_grid(y)[0], 
               norm=matplotlib.colors.Normalize(vmin=0.95, vmax=1.05),
               cmap='Greys');
    
