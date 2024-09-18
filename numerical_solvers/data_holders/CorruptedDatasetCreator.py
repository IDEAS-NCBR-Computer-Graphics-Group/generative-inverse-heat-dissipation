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
from numerical_solvers.data_holders.BlurringCorruptor import BlurringCorruptor
from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from scripts import datasets as ihd_datasets
from scripts.utils import save_png_norm
from configs.mnist.lbm_ns_config import get_lbm_ns_config
from configs.mnist.blurring_configs import get_blurr_config


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

# def corrupt_dataset(dataset, transform, output_dataset_dir, is_train_dataset):
#     solver_config = get_lbm_ns_config()
#     lbm_ns_Corruptor = LBM_NS_Corruptor(
#         solver_config,
#         transform=transform
#         )
#     lbm_ns_Corruptor._preprocess_and_save_data(
#         initial_dataset=dataset,
#         save_dir=output_dataset_dir,
#         is_train_dataset = is_train_dataset,
#         process_pairs = True,
#         process_all=True
#         )

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
    x, y = next(iter(original_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', y.shape)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
    
    # %% lbmize
    start = timer()
    process_all=True
    solver_config = get_lbm_ns_config()

    corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)

    corruptor = LBM_NS_Corruptor(
        solver_config,                                
        transform=transforms.Compose([torchvision.transforms.ToTensor()]))

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)    

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
    # %% blurr        
    start = timer()
    process_all=True
    solver_config = get_blurr_config()
    
    corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)
    
    corruptor = BlurringCorruptor(
        solver_config, 
        transform=transforms.Compose([torchvision.transforms.ToTensor()]))
    
    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = True,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)

    corruptor._preprocess_and_save_data(
        initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
        save_dir=corrupted_dataset_dir,
        is_train_dataset = False,
        process_pairs = solver_config.data.process_pairs,
        process_all=process_all)    

    # end = timer()
    print(f"Time in seconds: {end - start:.2f}")
    
    
    # %% see what you have done 
    
    # use same transform as in ihd code
    # transform = [
    #             torchvision.transforms.ToPILImage()
    #             transforms.Resize(28),
    #             transforms.CenterCrop(28),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor()
    #             ]
    # transform = transforms.Compose(transform)
    
    transform = None # the dataset is saved as torchtensor
    lbm_mnist_pairs = CorruptedDataset(train=is_train_dataset, 
                                       transform=transform, 
                                       target_transform=None, 
                                       load_dir=corrupted_dataset_dir)

    return corrupted_train_dataloader, corrupted_test_dataloader

def main(argv): 
    preprocess_dataset(FLAGS.config)

if __name__ == '__main__': 
    app.run(main)
    
    corrupted_dataloader = DataLoader(lbm_mnist_pairs, batch_size=32, shuffle=True)
    if solver_config.data.process_pairs:
        print(f"==processing pairs===")
        # x, (y, pre_y, corruption_amount, labels) = next(iter(corrupted_dataloader))
        # alternatively
        x, batch = ihd_datasets.prepare_batch(iter(corrupted_dataloader),'cpu')
        y, pre_y, corruption_amount, labels = batch
    else:
        x, (y, corruption_amount, labels) = next(iter(corrupted_dataloader))
    print('Input shape:', x.shape)
    print('batch_size = x.shape[0]:', x.shape[0])
    print('Labels:', labels)
    print('corruption_amount:', corruption_amount)

    # save_png_norm(current_file_path.parents[0], y, "test_norm.png") # test the plot saving fun
    
    
    clean_x, (noisy_x, less_noisy_x, corruption_amount, label) = next(iter(corrupted_dataloader))
    # x, (y, corruption_amount, label) = next(iter(test_dataloader))
    print('Input shape:', clean_x.shape)
    print('corruption_amount:', corruption_amount)
    print('batch_size = x.shape[0]:', clean_x.shape[0])
    print('Labels:', label.shape)
    # plt.imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys');
    # plt.imshow(torchvision.utils.make_grid(noisy_x, nrow=8)[0].clip(0.95, 1.05), cmap='Greys')
    # plt.imshow(torchvision.utils.make_grid(noisy_x)[0], cmap='Greys');

    fig, axs = plt.subplots(1, 3, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x')
    axs[1].set_title('noisy x')
    axs[2].set_title('less noisy x')

    # plt.imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys')
    # axs[0, 0].imshow(clean_x, cmap='Greys')
    axs[0].imshow(torchvision.utils.make_grid(clean_x)[0], cmap='Greys');
    axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0].clip(0.95, 1.05), cmap='Greys')
    axs[2].imshow(torchvision.utils.make_grid(less_noisy_x)[0].clip(0.95, 1.05), cmap='Greys')


# %%
