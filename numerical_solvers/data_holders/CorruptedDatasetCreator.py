from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
from torchvision import transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from pathlib import Path
import os
import logging


from configs.cifar10.lbm_ns_turb_config import get_lbm_ns_config
from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def corrupt_dataset(dataset, transform, output_dataset_dir, is_train_dataset):
    solver_config = get_lbm_ns_config()
    lbm_ns_Corruptor = LBM_NS_Corruptor(
        solver_config,
        transform=transform
        )
    lbm_ns_Corruptor._preprocess_and_save_data(
        initial_dataset=dataset,
        save_dir=output_dataset_dir,
        is_train_dataset = is_train_dataset,
        process_pairs = True,
        process_all=True
        )

def preprocess_dataset(config):
    # Get the chosen dataset
    trainloader, testloader = ihd_datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    # Determine the base folder (project root)
    # Adjust the number depending on your project structure
    current_file_path = Path(__file__).resolve()
    base_folder = current_file_path.parents[2]

    input_data_dir = os.path.join(base_folder, "data")
    dataset_name = f'corrupted_{config.data.dataset}'
    output_data_dir = os.path.join(input_data_dir, dataset_name)
    corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns_pair')

    transform = transforms.Compose([])

    # LBMize and save the dataset
    start = timer()
    logging.info(f"Corrupting {dataset_name} dataset.")
    logging.info("Corrupting train data.")
    corrupt_dataset(trainloader.dataset, transform, corrupted_dataset_dir, True)
    logging.info("Corrupting test data.")
    corrupt_dataset(testloader.dataset, transform, corrupted_dataset_dir, False)
    end = timer()
    logging.info(f"Corrupting took {end - start:.2f} seconds.")

    logging.info(f"Saving datset.") 
    transform = None
    lbm_train_pairs = CorruptedDataset(
        train=True, 
        transform=transform, 
        target_transform=None, 
        load_dir=corrupted_dataset_dir
        )
    corrupted_train_dataloader = DataLoader(lbm_train_pairs, batch_size=8, shuffle=True)
    lbm_test_pairs = CorruptedDataset(
        train=True, 
        transform=transform, 
        target_transform=None, 
        load_dir=corrupted_dataset_dir
        )
    corrupted_test_dataloader = DataLoader(lbm_test_pairs, batch_size=8, shuffle=True)

    return corrupted_train_dataloader, corrupted_test_dataloader

def main(argv): 
    preprocess_dataset(FLAGS.config)

if __name__ == '__main__': 
    app.run(main)
