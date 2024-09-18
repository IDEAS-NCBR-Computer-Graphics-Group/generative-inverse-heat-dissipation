from abc import ABC, abstractmethod

class BaseCorruptor(ABC):
    def __init__(self, transform=None, target_transform=None, save_dir='./corrupted_mnist'):

        self.transform = transform
        self.target_transform = target_transform
 
    @abstractmethod
    def _preprocess_and_save_data(self):
        pass