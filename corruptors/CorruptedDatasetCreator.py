from torchvision import transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from pathlib import Path
import logging
import os

from corruptors.BlurringCorruptor import BlurringCorruptor
from corruptors.LBM_NS_Corruptor import LBM_NS_Corruptor
from corruptors.CorruptedDataset import CorruptedDataset
from configs.cifar10.lbm_ns_turb_config import get_lbm_ns_config
from configs.cifar10.blurring_configs import get_blurr_config
from scripts import datasets as ihd_datasets

def corrupt_datasets(train, test, config, save_dir):
    dataset_name = f'corrupted_{config.data.dataset}'
    transform = transforms.Compose([])
    process_all = True

    fluid_save_dir = os.path.join(save_dir, 'fluid')
    logging.info(f"Fluid corruption on {dataset_name} dataset")
    start = timer()
    solver_config = get_lbm_ns_config()

    corruptor = LBM_NS_Corruptor(
        solver_config,                                
        transform=transform
    )

    logging.info("Fluid corruption on train split")
    corruptor._preprocess_and_save_data(
        initial_dataset=train.dataset,
        save_dir=fluid_save_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs
        )

    logging.info("Fluid corruption on test split")
    corruptor._preprocess_and_save_data(
        initial_dataset=train.dataset,
        save_dir=fluid_save_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs
        )    

    end = timer()
    logging.info(f"Fluid corruption took {end - start:.2f} seconds")

    blur_save_dir = os.path.join(save_dir, 'blur')
    logging.info(f"Blur corruption on {dataset_name} dataset")
    start = timer()
    solver_config = get_blurr_config()    
    
    corruptor = BlurringCorruptor(
        solver_config, 
        transform=transform
        )
    
    logging.info("Blur corruption on train split")
    corruptor._preprocess_and_save_data(
        initial_dataset=train.dataset,
        save_dir=blur_save_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs
        )

    logging.info("Blur corruption test split")
    corruptor._preprocess_and_save_data(
        initial_dataset=test.dataset,
        save_dir=blur_save_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs
        )    
    
    end = timer()
    logging.info(f"Blurring corruption took {end - start:.2f} seconds.")
    logging.info(f"Saving datasets.") 


def prepare_datasets(save_dir):
    logging.info(f"Preparing datasets.") 

    fluid_save_dir = os.path.join(save_dir, 'fluid')
    transform = None
    fluid_train_pairs = CorruptedDataset(
        train=True, 
        transform=transform, 
        target_transform=None, 
        load_dir=fluid_save_dir
        )
    fluid_train_dataloader = DataLoader(fluid_train_pairs, batch_size=8, shuffle=True)
    fluid_test_pairs = CorruptedDataset(
        train=False, 
        transform=transform, 
        target_transform=None, 
        load_dir=fluid_save_dir
        )
    fluid_test_dataloader = DataLoader(fluid_test_pairs, batch_size=8, shuffle=True)
    
    blur_save_dir = os.path.join(save_dir, 'blur')
    blur_train_pairs = CorruptedDataset(
        train=True, 
        transform=transform, 
        target_transform=None, 
        load_dir=blur_save_dir
        )
    blur_train_dataloader = DataLoader(blur_train_pairs, batch_size=8, shuffle=True)
    blur_test_pairs = CorruptedDataset(
        train=False, 
        transform=transform, 
        target_transform=None, 
        load_dir=blur_save_dir
        )
    blur_test_dataloader = DataLoader(blur_test_pairs, batch_size=8, shuffle=True)
    
    return (fluid_train_dataloader, fluid_test_dataloader), (blur_train_dataloader, blur_test_dataloader)


def preprocess_dataset(config):
    trainloader, testloader = ihd_datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    current_file_path = Path(__file__).resolve()
    base_folder = current_file_path.parents[2]

    input_data_dir = os.path.join(base_folder, "data")
    dataset_name = f'corrupted_{config.data.dataset}'
    output_data_dir = os.path.join(input_data_dir, dataset_name)
    save_dir = os.path.join(output_data_dir, 'lbm_ns_pair')

    corrupt_datasets(trainloader, testloader, config, save_dir)
    data = prepare_datasets(save_dir)
    
    return data
