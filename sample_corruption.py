from absl import flags
from absl import app
# from ml_collections.config_flags import config_flags

import os

from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm, save_png
from numerical_solvers.data_holders.CorruptedDatasetCreator import preprocess_dataset
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from scripts.utils import load_config_from_path

FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("config", None, "Path to the config file.")
flags.mark_flags_as_required(["config"])

def main(argv):
    produce_sample(FLAGS.config)
  
  
def produce_sample(config_path):
    config = load_config_from_path(config_path)
    trainloader, testloader = ihd_datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    storage_dir = 'runs'
    save_scriptname = 'corruption_samples'
    save_dir = os.path.join(storage_dir, save_scriptname)
    os.makedirs(save_dir, exist_ok=True)

    # # x, batch = next(iter(trainloader))
    clean_image, batch = ihd_datasets.prepare_batch(iter(trainloader),'cpu')
    corrupted_image, less_corrupted_image, corruption_amount, label = batch

    print('clean input shape:', clean_image.shape)
    print('corruption_amount:', corruption_amount)
    print('batch_size = x.shape[0]:', clean_image.shape[0])
    print('Labels:', label.shape)
    
    matplotlib.use('TkAgg')
    fig, axs = plt.subplots(1, 3, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x')
    axs[1].set_title('noisy x')
    axs[2].set_title('less noisy x')

    axs[0].imshow(torchvision.utils.make_grid(clean_image)[0], cmap='Greys')
    axs[1].imshow(torchvision.utils.make_grid(corrupted_image)[0], cmap='Greys')
    axs[2].imshow(torchvision.utils.make_grid(less_corrupted_image)[0], cmap='Greys')
    plt.savefig('output.png')
    plt.show()
    

if __name__ == '__main__':
    app.run(main)
    