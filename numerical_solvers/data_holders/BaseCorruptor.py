from abc import ABC, abstractmethod
import os
import shutil
import logging
class BaseCorruptor(ABC):
    def __init__(self, transform=None, target_transform=None, save_dir='./corrupted_mnist'):

        self.transform = transform
        self.target_transform = target_transform
 
    @abstractmethod
    def _preprocess_and_save_data(self, initial_dataset, save_dir, is_train_dataset: bool, process_pairs=False, process_all=True,  process_images=False):
        pass
    
    @abstractmethod
    def _corrupt(self, x, steps, generate_pair=False):
        pass
    
    def copy_train_dataset_as_test_dataset(self, save_dir):
        source_path = os.path.join(save_dir, "train_data.pt")
        destination_path = os.path.join(save_dir, "test_data.pt")
        if not os.path.exists(source_path):
            raise FileNotFoundError
        
        logging.warning(f"Using train dataset as test dataset.")
        if not os.path.exists(destination_path):   
            shutil.copy(source_path, destination_path) 
            logging.info(f"COPYING from {source_path} \n to {destination_path}")
        else:
            logging.info(f"test_data.pt already exists.")
  