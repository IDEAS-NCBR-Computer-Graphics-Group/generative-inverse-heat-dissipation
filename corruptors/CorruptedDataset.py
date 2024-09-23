import torch
from torch.utils.data import Dataset
import tqdm
import errno
import os

class CorruptedDataset(Dataset):
    def __init__(
            self,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            load_dir=None
            ):
        self.train = train
        self.transform = transform
        
        self.dataset_path = os.path.join(load_dir, 'train' if self.train else 'test')
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.dataset_path)    

    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, index):
        data_point_tensor = torch.load(os.path.join(self.dataset_path, f'data_point_{index}.pt'))

        data_length = len(data_point_tensor)-1

        if data_length == 4:
            original_image, modified_image, less_modified_image, corruption_amount, label = data_point_tensor

            if self.transform is not None:
                original_image = self.transform(original_image)
                modified_image = self.transform(modified_image)
                less_modified_image = self.transform(less_modified_image)

            return original_image, (modified_image, less_modified_image, corruption_amount, label)
      
        elif data_length == 5:
            original_image, modified_image, corruption_amount, label = data_point_tensor
            
            if self.transform is not None:
                original_image = self.transform(original_image)
                modified_image = self.transform(modified_image)

            return original_image, (modified_image, corruption_amount, label)

        else:
            raise ValueError(f"Unexpected data format in {data_point_tensor}, expected 4 or 5 elements, got {len(data_point_tensor)}")
