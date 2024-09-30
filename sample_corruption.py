from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import os
import torchvision
import matplotlib
from scripts import datasets as ihd_datasets
from scripts.utils import save_png
from corruptors.CorruptedDatasetCreator import preprocess_dataset
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def produce_sample(config):
    (fluid_train, fluid_test), (blur_train, blur_test) = preprocess_dataset(config)

    storage_dir = 'runs'
    save_scriptname = 'corruption_samples'
    save_dir = os.path.join(storage_dir, save_scriptname)
    os.makedirs(save_dir, exist_ok=True)

    x = ihd_datasets.prepare_batch(iter(blur_train),'cpu')





















    
    
    sampling_interval = 3

    num_images = len(x[0][0])
    fig, axs = plt.subplots((num_images + sampling_interval - 1) // sampling_interval, len(x[0]), 
                            sharex=True, sharey=True, figsize=(10, 10), facecolor='black')

    for i in range(len(x[0])):
        y = x[0][i]
        for j in range(0, num_images, sampling_interval):
            p = y[j].squeeze(0)

            subplot_row = j // sampling_interval
            axs[subplot_row, i].imshow(p, cmap='gray')
            axs[subplot_row, i].axis('off')

    plt.show()

def main(argv):
    produce_sample(FLAGS.config)
  
if __name__ == '__main__':
    app.run(main)
    