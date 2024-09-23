from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import os

from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm, save_png
from corruptors.CorruptedDatasetCreator import preprocess_dataset

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def produce_sample(config):
    (fluid_train, fluid_test), (blur_train, blur_test) = preprocess_dataset(config)

    storage_dir = 'runs'
    save_scriptname = 'corruption_samples'
    save_dir = os.path.join(storage_dir, save_scriptname)

    x, batch = ihd_datasets.prepare_batch(iter(fluid_train),'cpu')
    y, *_ = batch
    
    save_png_norm(save_dir, y, "post.png")
    save_png(save_dir, x, 'pre.png')

def main(argv):
    produce_sample(FLAGS.config)
  
if __name__ == '__main__':
    app.run(main)
    