# %% imports

import torch
import torchvision
import os
from pathlib import Path
from torch import nn
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# %% dataset

from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.data_holders.CorruptedDataset import CorruptedDataset
from configs.mnist.small_mnist_lbm_ns_turb_config import get_config as get_lbm_ns_turb_config

# %% figure out paths

print(f"Current working directory \t {os.getcwd()}")
current_file_path = Path(__file__).resolve()
# Determine the base folder (project root)
base_folder = current_file_path.parents[1]  # Adjust the number depending on your project structure
print(f"Base folder: {base_folder}")


input_data_dir = os.path.join(base_folder, "data")
output_data_dir = os.path.join(input_data_dir, 'corrupted_MNIST')
print(f"Input data folder: {input_data_dir}")

# %% lbmize

process_pairs=False
process_all=True
# solver_config = get_lbm_ns_config()
# corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns_pairs') if process_pairs else os.path.join(output_data_dir, 'lbm_ns')

solver_config = get_lbm_ns_turb_config()
corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)

start = timer()

lbm_ns_Corruptor = LBM_NS_Corruptor(
    solver_config,                                
    transform=transforms.Compose([torchvision.transforms.ToTensor()]))

lbm_ns_Corruptor._preprocess_and_save_data(
    initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
    save_dir=corrupted_dataset_dir,
    is_train_dataset = True,
    process_pairs = solver_config.data.process_pairs,
    process_all=True)

lbm_ns_Corruptor._preprocess_and_save_data(
    initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
    save_dir=corrupted_dataset_dir,
    is_train_dataset = False,
    process_pairs = solver_config.data.process_pairs,
    process_all=True)    

end = timer()
print(f"Data preprocessing time in seconds: {end - start:.2f}")


    
# %% dataloader
# Dataloader (you can mess with batch size)
batch_size = 128

trainDataset = CorruptedDataset(train=True, 
                                transform=None, # the dataset is saved as torchtensor
                                target_transform=None, 
                                load_dir=corrupted_dataset_dir)
train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

testDataset = CorruptedDataset(train=False, 
                               transform=None, # the dataset is saved as torchtensor
                               target_transform=None, 
                               load_dir=corrupted_dataset_dir)
test_dataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

x, (y, corruption_amount, label) = next(iter(train_dataloader))
# x, (y, corruption_amount, label) = next(iter(test_dataloader))
print('Input shape:', x.shape)
print('batch_size = x.shape[0]:', x.shape[0])
print('Labels:', label.shape)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
plt.imshow(torchvision.utils.make_grid(y, nrow=8)[0].clip(0.9, 1.1), cmap='Greys')
# plt.imshow(torchvision.utils.make_grid(y)[0], cmap='Greys');


# %% The model
net = UNet2DModel(
    sample_size=28,           # the target image resolution
    in_channels=1,            # the number of input channels, 3 for RGB images
    out_channels=1,           # the number of output channels
    layers_per_block=2,       # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64), # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",        # a regular ResNet downsampling block
        "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",          # a regular ResNet upsampling block
      ),
)

# print(net)

print(f"No of parameters: {sum([p.numel() for p in net.parameters()])}") # 1.7M vs the ~309k parameters of the BasicUNet
net.to(device)



# %%  Prepare the training loop

# Our loss finction
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Keeping a record of the losses for later viewing
losses = []

# How many runs through the data should we do?
n_epochs = 3

# Run training
print(f"batch_size={batch_size},\n" 
      f"no of batches={len(train_dataloader)},\n" 
      f"no of datapoints={len(train_dataloader.sampler)}")

start = timer()
for epoch in range(n_epochs):
    counter = 0 
    for clean_x, (noisy_x, corruption_amount, label) in train_dataloader:
        if counter % 50 == 0:
          print(f"batch counter = {counter}/{len(train_dataloader)}")
          
        counter = counter + 1
        # lbm_steps = torch.randint(0, 50, (batch_size,)).numpy()
        # noisy_x = lbm_corrupt(x, lbm_steps)
        # x = x.to(device)
        # noisy_x = noisy_x.to(device)
        
        # normal corrupt
        # Get some data and prepare the corrupted version
        # x = x.to(device) # Data on the GPU
        # noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
        # noisy_x = corrupt(x, noise_amount) # Create our noisy x
        # normal corrupt
  
        noisy_x = noisy_x.to(device)   
        clean_x = clean_x.to(device)     
        # Get the model prediction
        # pred = net(noisy_x, 0).sample #<<< Using timestep 0 always, adding .sample
        
        corruption_amount = corruption_amount.to(device)
        pred = net(noisy_x, corruption_amount).sample #<<< Using timestep 0 always, adding .sample

        # Calculate the loss
        loss = loss_fn(pred, clean_x) # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}/{n_epochs}. Average loss for this epoch: {avg_loss:05f}')


end = timer()
print(f"Training time in seconds: {end - start:.2f}")

# %% Save the trained model's state_dict
model_save_path =  os.path.join(current_file_path.parents[0], "unet_model.pth")

torch.save(net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# Load the state_dict into the model
net.load_state_dict(torch.load(model_save_path))
net.to(device)  # Send model to the appropriate device (GPU or CPU)
print(f"Model loaded from {model_save_path}")
# %% visualize training loss
# Losses
plt.plot(losses)
plt.yscale('log',base=10) 
plt.title(f'Loss over time')
plt.grid()
plt.show()


# %% Generate samples
fig, axs = plt.subplots(1, 3, figsize=(16, 10))

# Samples
n_steps = 1
# noisy_x = torch.rand(64, 1, 28, 28).to(device) # pure noise
# x, (noisy_x, corruption_amount, label) = next(iter(train_dataloader))
x, (noisy_x, corruption_amount, label) = next(iter(test_dataloader))

noisy_x = noisy_x.to(device)
corruption_amount = corruption_amount.to(device)
denoised_x = noisy_x.clone()     


for i in range(n_steps):
  # noise_amount = torch.ones((noisy_x.shape[0], )).to(device) * (1-(i/n_steps)) # Starting high going low
  with torch.no_grad():
    pred = net(denoised_x, corruption_amount).sample
  mix_factor = 1/(n_steps - i)
  denoised_x = denoised_x*(1-mix_factor) + pred*mix_factor

axs[0].imshow(torchvision.utils.make_grid(x, nrow=8)[0].clip(0, 1), cmap='Greys')
axs[0].set_title('Clean input');

axs[1].imshow(torchvision.utils.make_grid(denoised_x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Denoised');

axs[2].imshow(torchvision.utils.make_grid(noisy_x.detach().cpu(), nrow=8)[0].clip(0.9, 1.1), cmap='Greys')
axs[2].set_title('Noise to sample from');
# %%
