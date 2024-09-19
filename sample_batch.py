from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from timeit import default_timer as timer
from pathlib import Path

from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm, save_png
from numerical_solvers.data_holders.CorruptedDatasetCreator import preprocess_dataset
from matplotlib import pyplot as plt
import torchvision
import matplotlib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def main(config):
    (fluid_train, fluid_test), (blur_train, blur_test) = preprocess_dataset(FLAGS.config)
    save_dir = 'runs/test'

    process_pairs = True

    if process_pairs:
        print(f"==processing pairs===")
        # x, (y, pre_y, corruption_amount, labels) = next(iter(corrupted_dataloader))
        # alternatively
        x, batch = ihd_datasets.prepare_batch(iter(fluid_train),'cpu')
        y, pre_y, corruption_amount, labels = batch
    # else:
    #     x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', labels)
    print('corruption_amount:', corruption_amount)
    
    save_png_norm(save_dir, y, "test_norm.png")
    save_png(save_dir, x, 't0.png')


    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    # # plt.imshow(torchvision.utils.make_grid(y)[0], cmap='Greys');
    
    plt.imshow(torchvision.utils.make_grid(y)[0], 
               norm=matplotlib.colors.Normalize(vmin=0.95, vmax=1.05),
               cmap='Greys')

if __name__ == '__main__':
    app.run(main)
    