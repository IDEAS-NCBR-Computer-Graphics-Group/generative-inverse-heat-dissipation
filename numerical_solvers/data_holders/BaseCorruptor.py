import os
import warnings
from tqdm import tqdm
import torch

class BaseCorruptor():
    def __init__(self, transform=None, target_transform=None, save_dir='./corrupted_mnist'):

        self.transform = transform
        self.target_transform = target_transform

    def _preprocess_and_save_data(self, dataloader, save_dir, is_train_dataset: bool, process_pairs=False):
        """
        Preprocesses data and saves it to the specified directory.

        Args:
            initial_dataset (list): The initial dataset containing images and labels.
            save_dir (str): The directory to save the preprocessed data.
            process_pairs (bool): Flag indicating whether to process pairs of images (True) 
                                  or single corrupted images (False). Default is False.
        """
        initial_dataset = dataloader.dataset
        

        split = 'train' if is_train_dataset else 'test'

        split_save_dir = os.path.join(save_dir, split)
        if os.path.exists(split_save_dir):
            warnings.warn(f"[EXIT] Data not generated. Reason: file exist {save_dir} and is not empty.")
            return
        os.makedirs(split_save_dir)

        for i in tqdm(range(len(initial_dataset))):

            per_sample_corrupted_images = [] 
            per_sample_corruption_amounts = []
            labels = []
            pre_corrupted_images = [] if process_pairs else None     
            corruption_amount = 1

            image, label = initial_dataset[i]
            image = self.transform(image)

            if not process_pairs:
                per_sample_corrupted_images.append(image)
                per_sample_corruption_amounts.append(torch.tensor(0))
                labels.append(label)

            for k in range(1, int(self.max_steps)+1):

                corrupted_image, pre_corrupted_image = self._corrupt(
                    image,
                    corruption_amount,
                    generate_pair=process_pairs
                    )
                image = corrupted_image

                per_sample_corrupted_images.append(corrupted_image)
                per_sample_corruption_amounts.append(torch.tensor(k + corruption_amount))
                labels.append(label)

                if process_pairs:
                    pre_corrupted_images.append(pre_corrupted_image)

            file_path = os.path.join(split_save_dir, f'data_point_{i}.pt')

            data = torch.stack(per_sample_corrupted_images)
            corruption_amounts = torch.tensor(per_sample_corruption_amounts)
            labels = torch.tensor(labels)

            if process_pairs:
                pre_corrupted_images = torch.stack(pre_corrupted_images)
                torch.save((data, pre_corrupted_images, corruption_amounts, labels), file_path)
            else:
                torch.save((data, corruption_amounts, labels), file_path)