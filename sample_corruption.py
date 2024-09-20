from absl import flags
from absl import app
from ml_collections.config_flags import config_flags

from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm, save_png
from numerical_solvers.data_holders.CorruptedDatasetCreator import preprocess_dataset
from matplotlib import pyplot as plt
import torchvision
import matplotlib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def produce_sample(config):
    (fluid_train, fluid_test), (blur_train, blur_test) = preprocess_dataset(config)
    save_dir = 'runs/test'

    x, batch = ihd_datasets.prepare_batch(iter(fluid_train),'cpu')
    y, pre_y, corruption_amount, labels = batch
    
    save_png_norm(save_dir, y, "test_norm.png")
    save_png(save_dir, x, 't0.png')

def main(argv):
    produce_sample(FLAGS.config)
  
if __name__ == '__main__':
    app.run(main)
    