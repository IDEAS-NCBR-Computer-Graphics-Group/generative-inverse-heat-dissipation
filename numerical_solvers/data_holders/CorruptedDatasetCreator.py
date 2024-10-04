
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from pathlib import Path
import logging
import os

from numerical_solvers.data_holders.BlurringCorruptor import BlurringCorruptor
from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets

AVAILABLE_CORRUPTORS = {'fluid': LBM_NS_Corruptor, 'blurr': BlurringCorruptor}

def corrupt_datasets(train, test, config, save_dir):
    dataset_name = f'corrupted_{config.data.dataset}'

    logging.info(f"Corruption on {dataset_name} dataset")
    start = timer()

    corruptor = AVAILABLE_CORRUPTORS[config.solver.type](
        config=config,                                
        transform=config.data.transform
    )

    logging.info("Fluid corruption on train split")
    corruptor._preprocess_and_save_data(
        initial_dataset=train.dataset,
        save_dir=save_dir,
        process_all=False,
        is_train_dataset = True,
        process_pairs = config.data.process_pairs
        )

    logging.info("Fluid corruption on test split")
    corruptor._preprocess_and_save_data(
        initial_dataset=test.dataset,
        save_dir=save_dir,
        is_train_dataset = False,
        process_all = False,
        process_pairs = config.data.process_pairs
        )    

    end = timer()
    logging.info(f"Fluid corruption took {end - start:.2f} seconds")

def prepare_datasets(save_dir):
    
    logging.info(f"Preparing datasets.") 
    corrupted_train_dataset = CorruptedDataset(train=True, load_dir=save_dir)
    corrupted_test_dataset = CorruptedDataset(train=False, load_dir=save_dir)
    corrupted_train_dataloader = DataLoader(corrupted_train_dataset, batch_size=8, shuffle=True)
    corrupted_test_dataloader = DataLoader(corrupted_test_dataset, batch_size=8, shuffle=True)
    
    return corrupted_train_dataloader, corrupted_test_dataloader

def preprocess_dataset(trainloader, testloader, config):

    current_file_path = Path(__file__).resolve()
    base_folder = current_file_path.parents[2]

    input_data_dir = os.path.join(base_folder, "data")
    dataset_name = f'corrupted_{config.data.dataset}'
    output_data_dir = os.path.join(input_data_dir, dataset_name)
    save_dir = os.path.join(output_data_dir, f'{config.solver.type}_ns_pair')

    corrupt_datasets(trainloader, testloader, config, save_dir)
    data = prepare_datasets(save_dir)
    
    return data
