
# %% dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
import os, errno
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
from timeit import default_timer as timer

import sys
from pathlib import Path

from numerical_solvers.data_holders.BlurringCorruptor import BlurringCorruptor
from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm
from configs.mnist.lbm_ns_config import get_lbm_ns_config
from configs.mnist.blurring_configs import get_blurr_config


# sys.path.insert(0, '../../')
                                
# %% dataset
# Create the DiffusedMNIST dataset
if __name__ == '__main__': 
    print(f"Current working directory \t {os.getcwd()}")
    current_file_path = Path(__file__).resolve()
    # Determine the base folder (project root)
    base_folder = current_file_path.parents[2]  # Adjust the number depending on your project structure
    print(f"Base folder: {base_folder}")
    
    is_train_dataset = True

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
    process_all=True
    solver_config = get_lbm_ns_config()

    corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)

    corruptor = LBM_NS_Corruptor(
        solver_config,                                
        transform=transforms.Compose([torchvision.transforms.ToTensor()]))

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)    

    end = timer()
    print(f"Time in seconds: {end - start:.2f}")

    # %% blurr        
    start = timer()
    process_all=True
    solver_config = get_blurr_config()
    
    corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)
    
    corruptor = BlurringCorruptor(
        solver_config, 
        transform=transforms.Compose([torchvision.transforms.ToTensor()]))
    
    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)    

    end = timer()
    print(f"Time in seconds: {end - start:.2f}")
    
    
    # %% see what you have done 
    
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



    
    corrupted_dataloader = DataLoader(lbm_mnist_pairs, batch_size=32, shuffle=True)
    if solver_config.data.process_pairs:
        print(f"==processing pairs===")
        # x, (y, pre_y, corruption_amount, labels) = next(iter(corrupted_dataloader))
        # alternatively
        x, batch = ihd_datasets.prepare_batch(iter(corrupted_dataloader),'cpu')
        y, pre_y, corruption_amount, labels = batch
    else:
        x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', labels)
    print('corruption_amount:', corruption_amount)

    # save_png_norm(current_file_path.parents[0], y, "test_norm.png") # test the plot saving fun
    
    
    clean_x, (noisy_x, less_noisy_x, corruption_amount, label) = next(iter(corrupted_dataloader))
    # x, (y, corruption_amount, label) = next(iter(test_dataloader))
    print('Input shape:', clean_x.shape)
    print('corruption_amount:', corruption_amount)
    print('batch_size = x.shape[0]:', clean_x.shape[0])
    print('Labels:', label.shape)
    # plt.imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys');
    # plt.imshow(torchvision.utils.make_grid(noisy_x, nrow=8)[0].clip(0.95, 1.05), cmap='Greys')
    # plt.imshow(torchvision.utils.make_grid(noisy_x)[0], cmap='Greys');

    fig, axs = plt.subplots(1, 3, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x')
    axs[1].set_title('noisy x')
    axs[2].set_title('less noisy x')

    # plt.imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys')
    # axs[0, 0].imshow(clean_x, cmap='Greys')
    axs[0].imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys');
    axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0].clip(0.95, 1.05), cmap='Greys')
    axs[2].imshow(torchvision.utils.make_grid(less_noisy_x)[0].clip(0.95, 1.05), cmap='Greys')


# %%
