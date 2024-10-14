from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from scripts import datasets as ihd_datasets
from scripts import sampling, utils
from numerical_solvers.data_holders.CorruptedDatasetCreator import AVAILABLE_CORRUPTORS

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def main(argv):
    produce_sample(FLAGS.config)
  
  
def produce_sample(config):
    torch.manual_seed(config.seed)
    np.random.seed(2021)

    trainloader, testloader = ihd_datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    storage_dir = 'runs'
    save_scriptname = 'corruption_samples'
    save_dir = os.path.join(storage_dir, save_scriptname)
    os.makedirs(save_dir, exist_ok=True)
    
    clean_image, batch = ihd_datasets.prepare_batch(iter(trainloader),'cpu')
    corrupted_image, less_corrupted_image, corruption_amount, label = batch

    print('clean input shape:', clean_image.shape)
    print('corruption_amount:', corruption_amount)
    print('batch_size = x.shape[0]:', clean_image.shape[0])
    print('Labels:', label.shape)
    
    matplotlib.use('TkAgg')
    fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x', fontsize=24)
    axs[1].set_title('noisy x', fontsize=24)
    axs[2].set_title('less noisy x', fontsize=24)

    axs[0].imshow(torchvision.utils.make_grid(clean_image)[0], cmap='Greys')
    axs[1].imshow(torchvision.utils.make_grid(corrupted_image)[0], cmap='Greys')
    axs[2].imshow(torchvision.utils.make_grid(less_corrupted_image)[0], cmap='Greys')
    plt.savefig(os.path.join(save_dir, 'Corruption_pairs_sample.png'), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    
    corruptor=AVAILABLE_CORRUPTORS[config.solver.type](
        config=config,
        transform=config.data.transform)

    
    clean_image, batch = ihd_datasets.prepare_batch(iter(trainloader),'cpu')
    noisy_sample = torch.empty_like(clean_image)
  
    for index in range(clean_image.shape[0]):
        tmp, _ = corruptor._corrupt(clean_image[index], config.solver.n_denoising_steps)
        noisy_sample[index] = tmp
        
    
    fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x', fontsize=24)
    axs[1].set_title('noisy x', fontsize=24)

    axs[0].imshow(torchvision.utils.make_grid(clean_image)[0], cmap='Greys')
    axs[1].imshow(torchvision.utils.make_grid(noisy_sample)[0], cmap='Greys')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fully_corrupted_sample.png'), bbox_inches='tight')
    # plt.show()
    plt.close()

    n_denoising_steps = 200
    initial_corrupted_sample, clean_initial_sample, intermediate_corruption_samples = sampling.get_initial_corrupted_sample(
        trainloader, n_denoising_steps, corruptor)
    
    utils.save_gif(save_dir, intermediate_corruption_samples, "corruption_init.gif")
    utils.save_video(save_dir, intermediate_corruption_samples, filename="corruption_init.mp4")
    utils.save_png(save_dir, clean_initial_sample, "clean_init.png")
    
if __name__ == '__main__':
    app.run(main)
