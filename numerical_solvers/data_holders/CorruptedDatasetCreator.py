# import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# , Dataset
# from PIL import Image, ImageFilter
import os
# , errno
import torchvision
# import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
from timeit import default_timer as timer

# import sys
from pathlib import Path

from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm
from configs.mnist.lbm_ns_turb_config import get_lbm_ns_config

def corrupt_dataset(original_dataset, processed_dataset_path, transform, is_train_dataset=True, process_pairs=True, process_all=True):

    solver_config = get_lbm_ns_config()
    lbm_ns_Corruptor = LBM_NS_Corruptor(
        solver_config,                                
        transform=transform)
    
    lbm_ns_Corruptor._preprocess_and_save_data(
        initial_dataset = original_dataset,
        save_dir = processed_dataset_path,
        is_train_dataset = is_train_dataset,
        process_pairs = process_pairs,
        process_all = process_all)

def preprocess(input_data_dir, output_data_dir, is_train_dataset=True, process_pairs=True, debug=False):
    if debug:
        print(f"Current working directory \t {os.getcwd()}")
        print(f"Base folder: {base_folder}")
    
    current_file_path = Path(__file__).resolve()
    base_folder = current_file_path.parents[2]  # depending on your project structure
    
    # https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
    transform = transforms.Compose([torchvision.transforms.ToTensor()])
    original_dataloader = DataLoader(torchvision.datasets.MNIST(root=input_data_dir, 
                                    train=is_train_dataset, download=True, transform=transform), 
                                    batch_size=8, shuffle=True)

    x, y = next(iter(original_dataloader))

    if debug:
        print('Input shape:', x.shape)
        print('batch_size = x.shape[0]:', x.shape[0])
        print('Labels:', y.shape)
        plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');

    corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns_pair')

    initial_dataset = datasets.MNIST(root=input_data_dir, train=is_train_dataset, download=True)

    start = timer()
    corrupt_dataset(
        initial_dataset,
        corrupted_dataset_dir,
        transform,
        process_pairs=process_pairs
        )
    end = timer()
    if debug:
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
        if debug:
            print(f"\nProcessing pairs\n")
        # x, (y, pre_y, corruption_amount, labels) = next(iter(corrupted_dataloader))
        # alternatively
        x, batch = ihd_datasets.prepare_batch(iter(corrupted_dataloader),'cpu')
        y, pre_y, corruption_amount, labels = batch
    else:
        x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    
    if debug:
        print('Input shape:', x.shape)
        print('batch_size = x.shape[0]:', x.shape[0])
        print('Labels:', labels)
        print('corruption_amount:', corruption_amount)
        save_png_norm(
            current_file_path.parents[0],
            y,
            "test_norm.png")
        plt.imshow(
            torchvision.utils.make_grid(x)[0],
            cmap='Greys'
            )
        plt.imshow(
            torchvision.utils.make_grid(y)[0], 
            norm=matplotlib.colors.Normalize(vmin=0.95, vmax=1.05),
            cmap='Greys'
            )

def main():
    input_data_dir = os.path.join(base_folder, "data")
    output_data_dir = os.path.join(input_data_dir, dataset_name)
    preprocess(
        input_data_dir,
        output_data_dir
    )

if __name__ == '__main__': 
    main()