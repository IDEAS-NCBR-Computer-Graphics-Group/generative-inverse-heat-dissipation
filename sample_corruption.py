from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import os

from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm, save_png
from numerical_solvers.data_holders.CorruptedDatasetCreator import preprocess_dataset

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def produce_sample(config):
    trainloader, testloader = ihd_datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    storage_dir = 'runs'
    save_scriptname = 'corruption_samples'
    save_dir = os.path.join(storage_dir, save_scriptname)
    os.makedirs(save_dir, exist_ok=True)

    # # x, batch = next(iter(trainloader))
    # x, batch = ihd_datasets.prepare_batch(iter(trainloader),'cpu')
    # y, a, h = batch

    # # print(y.shape)    
    # print(x.shape)
    # print(y.shape)
    # print(a.shape)
    # print(h)

    # save_png(save_dir, x, 'pre.png')
    # save_png(save_dir, y, "post.png")

def main(argv):
    produce_sample(FLAGS.config)
  
if __name__ == '__main__':
    app.run(main)
    