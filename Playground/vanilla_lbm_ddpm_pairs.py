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

from numerical_solvers.corruptors.GaussianBlurringCorruptor import GaussianBlurringCorruptor
from numerical_solvers.corruptors.LBM_NS_Corruptor import LBM_NS_Corruptor
from numerical_solvers.corruptors.CorruptedDataset import CorruptedDataset
# from configs.mnist.small_mnist_lbm_ns_config import get_config as lbm_ns_config
from configs.mnist.small_mnist_gaussian_blurring_config import get_config as get_small_mnist_config
# from configs.mnist.small_mnist_lbm_ns_turb_config import get_lbm_ns_config as get_lbm_ns_turb_config

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
start = timer()
process_all=True

# config = lbm_ns_config()
# corrupted_dataset_dir = os.path.join(output_data_dir, solver_config.data.processed_filename)

# config = get_lbm_ns_turb_config()
# corrupted_dataset_dir = os.path.join(output_data_dir, 'lbm_ns_turb')


# corruptor = LBM_NS_Corruptor(
#     config,                                
#     transform=transforms.Compose([torchvision.transforms.ToTensor()]))

# corruptor._preprocess_and_save_data(
#     initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
#     save_dir=corrupted_dataset_dir,
#     is_train_dataset = True,
#     process_pairs = config.data.process_pairs,
#     process_all=True)

# corruptor._preprocess_and_save_data(
#     initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
#     save_dir=corrupted_dataset_dir,
#     is_train_dataset = False,
#     process_pairs = config.data.process_pairs,
#     process_all=True)    

config = get_small_mnist_config()
corrupted_dataset_dir = os.path.join(output_data_dir, config.data.processed_filename)

corruptor = GaussianBlurringCorruptor(
    config, 
    transform=transforms.Compose([torchvision.transforms.ToTensor()]))

corruptor._preprocess_and_save_data(
    initial_dataset=datasets.MNIST(root=input_data_dir, train=True, download=True),
    save_dir=corrupted_dataset_dir,
    is_train_dataset = True,
    process_pairs = config.data.process_pairs,
    process_all=True)

corruptor._preprocess_and_save_data(
    initial_dataset=datasets.MNIST(root=input_data_dir, train=False, download=True),
    save_dir=corrupted_dataset_dir,
    is_train_dataset = False,
    process_pairs = config.data.process_pairs,
    process_all=True) 
    
end = timer()
print(f"Data preprocessing time in seconds: {end - start:.2f}")

    
# %% dataloader
# Dataloader (you can mess with batch size)
training_batch_size = 128

trainDataset = CorruptedDataset(train=True, 
                                transform=None, # the dataset is saved as torchtensor
                                target_transform=None, 
                                load_dir=corrupted_dataset_dir)
train_dataloader = DataLoader(trainDataset, batch_size=training_batch_size, shuffle=True)

testDataset = CorruptedDataset(train=False, 
                               transform=None, # the dataset is saved as torchtensor
                               target_transform=None, 
                               load_dir=corrupted_dataset_dir)
test_dataloader = DataLoader(testDataset, batch_size=8, shuffle=True)

clean_x, (blurred_x, less_blurred_x, corruption_amount, label) = next(iter(train_dataloader))
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
# axs[1].imshow(torchvision.utils.make_grid(noisy_x)[0].clip(0.95, 1.05), cmap='Greys')
# axs[2].imshow(torchvision.utils.make_grid(less_noisy_x)[0].clip(0.95, 1.05), cmap='Greys')

axs[1].imshow(torchvision.utils.make_grid(blurred_x)[0].clip(config.solver.min_init_gray_scale, config.solver.max_init_gray_scale), cmap='Greys')
axs[2].imshow(torchvision.utils.make_grid(less_blurred_x)[0].clip(config.solver.min_init_gray_scale, config.solver.max_init_gray_scale), cmap='Greys')


# %% The model

# 
# net = UNet2DModel(
#     sample_size=28,           # the target image resolution
#     in_channels=1,            # the number of input channels, 3 for RGB images
#     out_channels=1,           # the number of output channels
#     layers_per_block=2,       # how many ResNet layers to use per UNet block
#     block_out_channels=(32, 64, 64), # Roughly matching our basic unet example
#     down_block_types=(
#         "DownBlock2D",        # a regular ResNet downsampling block
#         "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
#         "AttnDownBlock2D",
#     ),
#     up_block_types=(
#         "AttnUpBlock2D",
#         "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
#         "UpBlock2D",          # a regular ResNet upsampling block
#       ),
# )

model_save_path =  os.path.join(current_file_path.parents[0], "unet_model_pairs.pth")

# use the same network as in ihd paper
from model_code import utils as mutils
from configs.mnist.small_mnist import get_config as get_small_mnist_config
small_mnist_config = get_small_mnist_config()
net = mutils.create_model(config)
print(net)

print(f"No of parameters: {sum([p.numel() for p in net.parameters()])}") # 1.7M vs the ~309k parameters of the BasicUNet
net.to(device)



# %%  Prepare the training loop

# Our loss finction
loss_fn = nn.MSELoss()

# The optimizer
# opt = torch.optim.Adam(net.parameters(), lr=1e-4)

from scripts import losses
opt = losses.get_optimizer(config, net.parameters())
# Keeping a record of the losses for later viewing
losses = []

# How many runs through the data should we do?
n_epochs = 5

# %% Run training
print(f"batch_size={training_batch_size},\n" 
      f"no of batches={len(train_dataloader)},\n" 
      f"no of datapoints={len(train_dataloader.sampler)}")

start = timer()

def add_noise(x, amount):
    # https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb#scrollTo=crLhiM4xMRoZ
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1.-amount) + noise*amount 

sigma = 0.01 # traing noise
delta = sigma*1.25 # sampling noise

for epoch in range(n_epochs):
    counter = 0 
    for clean_x, (blurred_x, less_blurred_x, corruption_amount, label) in train_dataloader:
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
  
        blurred_x = blurred_x.to(device)   
        less_blurred_x = less_blurred_x.to(device)
        clean_x = clean_x.to(device)     
        # Get the model prediction
        # pred = net(noisy_x, 0).sample #<<< Using timestep 0 always, adding .sample
        
        corruption_amount = corruption_amount.to(device)
        # pred = net(noisy_x, corruption_amount).sample #<<< Using timestep 0 always, adding .sample
        
        # noise = torch.randn_like(blurred_x) * sigma
        # perturbed_data = blurred_x + noise # add training noise
        # diff = net(perturbed_data, corruption_amount).sample #<<< Using timestep 0 always, adding .sample
        # prediction = perturbed_data + diff #instead of less noisy learn the diff
        
        noise = torch.randn_like(blurred_x) * sigma
        perturbed_data = blurred_x + noise
        diff = net(perturbed_data, corruption_amount)
        prediction = perturbed_data + diff
        
        # Calculate the loss
        loss = loss_fn(less_blurred_x, prediction) # How close is the output to the true 'less_noisy_x'?

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

torch.save(net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# %%  Load the state_dict into the model
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


# %% Generate noisy samples


# n_steps = int(solver_config.solver.max_steps) # 5
n_denoising_steps = fwd_steps = 50
# noisy_x = torch.rand(64, 1, 28, 28).to(device) # pure noise
# x, (noisy_x, less_noisy_x, corruption_amount, label) = next(iter(test_dataloader))
clean_x, (_, _, _, _) = next(iter(test_dataloader))

blurred_x = torch.empty_like(clean_x)
for index in range(clean_x.shape[0]):
    tmp, _ = corruptor._corrupt(clean_x[index], fwd_steps) # blur to the max level
    blurred_x[index] = tmp

step_history = [blurred_x.detach().cpu()]
pred_output_history = []
blurred_x = blurred_x.to(device)

deblurred_x = blurred_x.clone() 

print("corruption_amount[0].item(), n_steps, i")

# denoise samples
with torch.no_grad():
    for i in range(n_denoising_steps, 0, -1):
        vec_fwd_steps = torch.ones(blurred_x.shape[0], device=device, dtype=torch.long) * i
        # diff = net(deblurred_x, vec_fwd_steps).sample
        diff = net(deblurred_x, vec_fwd_steps)
        
        deblurred_x = deblurred_x + diff
        
        # add sampling noise
        noise = torch.randn_like(deblurred_x)
        deblurred_x = deblurred_x + noise*delta 
        
        pred_output_history.append(diff.detach().cpu())
        step_history.append(deblurred_x.detach().cpu())
        print(vec_fwd_steps[0].item(), n_denoising_steps, i)

fig, axs = plt.subplots(1, 3, figsize=(20, 16))
axs[0].imshow(torchvision.utils.make_grid(clean_x, nrow=8)[0].clip(0, 1), cmap='Greys')
axs[0].set_title('Clean input');

axs[1].imshow(torchvision.utils.make_grid(deblurred_x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Denoised');

axs[2].imshow(torchvision.utils.make_grid(blurred_x.detach().cpu(), nrow=8)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Noise to sample from');
plt.show()
plt.close()

fig, axs = plt.subplots(n_denoising_steps, 2, figsize=(20, 25), sharex=True)
axs[0,0].set_title('x (model input)')
axs[0,1].set_title('NN prediction')
for i in range(n_denoising_steps):
    axs[i, 0].imshow(torchvision.utils.make_grid(
        step_history[i])[0].clip(config.solver.min_init_gray_scale, 
                                 config.solver.max_init_gray_scale), cmap='Greys')
    axs[i, 1].imshow(torchvision.utils.make_grid(
        pred_output_history[i])[0].clip(
            config.solver.min_init_gray_scale, 
            config.solver.max_init_gray_scale), cmap='Greys')


# %%
