import torch
import os
import logging
import cv2
from pathlib import Path
from PIL import Image
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image
from model_code import utils as mutils
from scripts import losses
from model_code.ema import ExponentialMovingAverage
import pickle
import importlib.util

from ml_collections.config_dict import ConfigDict

def setup_logging(workdir):
    # Set up file logging after workdir is available
    os.makedirs(workdir, exist_ok=True)
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger('')
    root_logger.handlers = []  # Clear existing handlers

    # Set up the format you want, including the line number
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(workdir, 'training.log'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Set up console logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Set logging level for third-party libraries like absl or TensorFlow
    logging.getLogger('absl').setLevel(logging.WARNING)  # Suppress absl logging if applicable


def load_config_from_path(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Assuming the config file defines a function get_config()
    config_data = config_module.get_config()
    
    # Convert to ConfigDict
    return ConfigDict(config_data)

def restore_checkpoint(ckpt_dir, state, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['step'] = loaded_state['step']
        if 'ema' in state:
            state['ema'].load_state_dict(loaded_state['ema'])
        return state


def load_model(ckpt_dir, model, device):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if not os.path.exists(ckpt_dir):
        Path(os.path.dirname(ckpt_dir)).mkdir(parents=True, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return None
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        return loaded_state['model']


def save_checkpoint(ckpt_dir, state):
    """Taken from https://github.com/yang-song/score_sde_pytorch"""
    if 'ema' in state:
        saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'step': state['step'],
            'ema': state['ema'].state_dict()
        }
    else:
        saved_state = {
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)


def save_video(save_dir, samples):
    """ Saves a video from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the video"""
    padding = 0
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        sample = samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy(
        ).transpose(1, 2, 0).astype(np.uint8)
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        imgs.append(image_grid)
    #video_size = tuple(reversed(tuple(5*s for s in imgs[0].shape[:2])))
    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    writer = cv2.VideoWriter(os.path.join(save_dir, "process.mp4"), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, video_size)
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                           fy=0, interpolation=cv2.INTER_CUBIC)
        writer.write(image)
    writer.release()


def save_gif(save_dir, samples, name="process.gif"):
    """ Saves a gif from Pytorch tensor 'samples'. Arguments:
    samples: Tensor of shape: (video_length, n_channels, height, width)
    save_dir: Directory where to save the gif"""
    nrow = int(np.sqrt(samples[0].shape[0]))
    imgs = []
    for idx in range(len(samples)):
        s = samples[idx].cpu().detach().numpy()[:36]
        s = np.clip(s * 255, 0, 255).astype(np.uint8)
        image_grid = make_grid(torch.Tensor(s), nrow, padding=2)
        im = Image.fromarray(image_grid.permute(
            1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)
    imgs[0].save(os.path.join(save_dir, name), save_all=True,
                 append_images=imgs[1:], duration=0.5, loop=0)


def save_tensor(save_dir, data, name):
    """ Saves a Pytorch Tensor to save_dir with the given name."""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, data.cpu().numpy())


def save_number(save_dir, data, name):
    """ Saves the number in argument 'data' as a text file and a .np file."""
    with open(os.path.join(save_dir, name), "w") as fout:
        fout.write(str(data))
    with open(os.path.join(save_dir, name) + ".np", "wb") as fout:
        np.save(fout, data)


def save_tensor_list(save_dir, data_list, name):
    """Saves a list of Pytorch tensors to save_dir with name 'name'"""
    with open(os.path.join(save_dir, name), "wb") as fout:
        np.save(fout, np.array([d.cpu().detach().numpy() for d in data_list]))


def save_png(save_dir, data, name, nrow=None):
    """Save tensor 'data' as a PNG"""
    if nrow == None:
        nrow = int(np.sqrt(data.shape[0]))
    image_grid = make_grid(data, nrow, padding=2)
    with open(os.path.join(save_dir, name), "wb") as fout:
        save_image(image_grid, fout)

def save_png_norm(save_dir, data, name, nrow=None):
    """Save tensor 'data' as a PNG with specified normalization and colormap."""
    if nrow is None:
        nrow = int(np.sqrt(data.shape[0]))
        
    image_grid = torchvision.utils.make_grid(data, nrow, padding=2)[0]

    # # Convert the tensor to a NumPy array
    # image_np = image_grid.cpu().numpy()
    # # Transpose from (C, H, W) to (H, W, C)
    # image_np = np.transpose(image_np, (1, 2, 0))

    image_grid = torchvision.utils.make_grid(data)[0]
    image_np = image_grid.cpu().numpy()
    # Apply normalization
    norm = matplotlib.colors.Normalize(vmin=0.95, vmax=1.05)
    image_np_norm = norm(image_np)

    # Apply the 'Greys' colormap
    cmap = plt.get_cmap('Greys')
    image_colored = cmap(image_np_norm)

    # plt.imshow(image_colored, cmap=cmap);
    # Save the image
    # with open(os.path.join(save_dir, name), "wb") as fout:
    plt.imsave(os.path.join(save_dir, name), image_colored)
    

def append_to_log_file(dataset, experiment, additional_param, score_name, score_value):
    """
    Helper function for saving evaluation results to a .txt file. Arguments: 
    dataset: The dataset (e.g., cifar10, ffhq, ...)
    experiment: Which experiment in 'runs/dataset/evaluation_results'? E.g. 'delta_sweep' or something
    additional_param: How to distinguish the result from other saved results in the experiment?
    score_name: E.g., 'FID', 'ELBO'
    score_value: The value to be saved
    """
    experiment_folder = os.path.join(
        "runs", dataset, "evaluation_results", experiment)
    try:
        os.makedirs(experiment_folder)
    except OSError:
        logging.info("Directory already created")

    log_file = os.path.join(experiment_folder, "{}_log.txt".format(score_name))
    with open(log_file, "a") as f:
        f.write("{}: {}\n".format(additional_param, score_value))
    logging.info("Written to log file")


def append_to_dict(dataset, experiment, additional_param, score_name, score_value):
    """
    Helper function for saving evaluation results to a pickled file with a Python dict.
    Arguments: 
    dataset: The dataset (e.g., cifar10, ffhq, ...)
    experiment: Which experiment in 'runs/dataset/evaluation_results'? E.g. 'delta_sweep' or something
    additional_param: How to distinguish the result from other saved results in the experiment?
    score_name: E.g., 'FID', 'ELBO'
    score_value: The value to be saved
    """
    experiment_folder = os.path.join(
        "runs", dataset, "evaluation_results", experiment)
    try:
        os.makedirs(experiment_folder)
    except OSError:
        logging.info("Directory already created")

    dict_file = os.path.join(
        experiment_folder, "{}_dict.pkl".format(score_name))
    if os.path.exists(dict_file) and not os.stat(dict_file).st_size == 0:
        with open(dict_file, "rb") as f:
            dict_ = pickle.load(f)
        with open(dict_file, "wb") as f:
            dict_[additional_param] = score_value
            pickle.dump(dict_, f)
    else:
        with open(dict_file, "wb") as f:
            dict_ = {additional_param: score_value}
            pickle.dump(dict_, f)
    logging.info("Written to the dict file")


def load_model_from_checkpoint(config, checkpoint_dir, checkpoint):
    """A wrapper around restore_checkpoint() to load a model 
            with EMA from a checkpoint folder, and then discard the EMA and optimizer
            part of the state
    """
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    checkpoint_path = os.path.join(
        checkpoint_dir, 'checkpoint_{}.pth'.format(checkpoint))
    state = restore_checkpoint(checkpoint_path, state, device=config.device)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    model = state['model']
    state['ema'].copy_to(model.parameters())
    return model


def load_model_from_checkpoint_dir(config, checkpoint_dir):
    """Another input definition for the restore_checkpoint wrapper,
    without a specified checkpoint number. 
    Assumes that the folder has file "checkpoint.pth"
    """
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    state = restore_checkpoint(checkpoint_path, state, device=config.device)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    model = state['model']
    state['ema'].copy_to(model.parameters())
    return model
