
from absl import flags
from absl import app
import os, shutil
from pathlib import Path
import logging
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import taichi as ti
import cv2

from scripts import sampling, utils, datasets
from numerical_solvers.corruptors.CorruptedDatasetCreator import AVAILABLE_CORRUPTORS
from scripts.utils import load_config_from_path
from torchvision.utils import make_grid

FLAGS = flags.FLAGS

def main(argv):
    # Example
    # python sample_corruption.py --config=configs/ffhq/ffhq_128_lbm_ns_config_high_visc.py
    # python sample_corruption.py --config=configs/mnist/small_mnist_lbm_ns_config.py
    if FLAGS.demo:
        live_demo(FLAGS.config)
    else:
        produce_fwd_sample(FLAGS.config)
  
def live_demo(config):
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    
    config = load_config_from_path(config)
    if getattr(config, 'solver', None): 
        solver = config.solver
        config.solver = None
        
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainloader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        train_batch_size=config.eval.batch_size
        )
    
    config.solver = solver
    corruptor=AVAILABLE_CORRUPTORS[config.solver.type](
        config=config,
        transform=config.data.transform
    )

    n_denoising_steps = config.solver.n_denoising_steps
    original_pil_image, _ = next(iter(trainloader))
    noisy_initial_images = original_pil_image.clone()
    intermediate_samples = []

    for index in range(original_pil_image.shape[0]):
        noisy_initial_images[index], _ = corruptor._corrupt(original_pil_image[index], n_denoising_steps)
        intermediate_samples.append(corruptor.intermediate_samples)
        corruptor.solver.turbulenceGenerator.randomize()
    intermediate_samples = [torch.stack([sample[i] for sample in intermediate_samples]) for i in range(len(intermediate_samples[0]))]

    padding = 0
    nrow = int(np.sqrt(intermediate_samples[0].shape[0]))
    imgs = []
    for idx in range(len(intermediate_samples)):
        sample = intermediate_samples[idx].cpu().detach().numpy()
        sample = np.clip(sample * 255, 0, 255)
        image_grid = make_grid(torch.Tensor(sample), nrow, padding=padding).numpy(
        ).transpose(1, 2, 0).astype(np.uint8)
        imgs.append(image_grid)
    video_size = tuple(reversed(tuple(s for s in imgs[0].shape[:2])))
    images = []
    for i in range(len(imgs)):
        image = cv2.resize(imgs[i], video_size, fx=0,
                        fy=0, interpolation=cv2.INTER_CUBIC)
        image = np.ascontiguousarray(np.flip(np.transpose(image.T, [1,2,0]),axis=1))
        images.append(image.astype(np.float32)/255.0)

    window = ti.ui.Window('CG - Renderer', res=image.shape[:2])
    canvas = window.get_canvas()
    index = 0
    while window.running:
        window.GUI.begin("Display Panel", 0, 0, 0.4, 0.2)
        index = window.GUI.slider_int("corrupt_idx", index, 0, len(intermediate_samples)-1)
        if window.GUI.button("+1"): index += 1
        if window.GUI.button("-1"): index -= 1
        window.GUI.end()
        canvas.set_image(images[index])
        window.show()
  
def produce_fwd_sample(config_path):
    config = utils.load_config_from_path(config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # save_dir = os.path.join("runs", "sample_corruption", f"caseId_{config.stamp.fwd_solver_hash}")
    save_dir = os.path.join("tests", "artifacts", f"fwd_corruption_{config.stamp.fwd_solver_hash}")
    utils.setup_logging(save_dir)
    logging.info(f"save_dir: {save_dir}")
    logging.info(f"config.corrupt_sched has {len(config.solver.corrupt_sched)} elements:\n {config.solver.corrupt_sched}")


    # Get the absolute path of the current script
    current_file_path = Path(__file__).resolve()
    project_dir = current_file_path.parents[0]

    # remove previous corrupted dataset
    dataset_dir = utils.get_save_dir(project_dir, config)
    logging.info(f"dataset_dir: {dataset_dir}")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        logging.info(f"Removed {dataset_dir}")

    trainloader, testloader = datasets.get_dataset(config,
                                                        uniform_dequantization=config.data.uniform_dequantization)

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, save_dir)

    default_cfg_path = os.path.join(*config_path.split(os.sep)[0:2], f'default_lbm_{config.data.dataset.lower()}_config.py')
    if os.path.isfile(default_cfg_path):
        shutil.copy2(default_cfg_path, save_dir)

    clean_image, batch = datasets.prepare_batch(iter(trainloader), 'cpu')
    corrupted_image, less_corrupted_image, corruption_amount, label = batch

    logging.info(f"clean input shape: {clean_image.shape}")
    logging.info(f"corruption_amount: {corruption_amount}")
    logging.info(f"batch_size = x.shape[0]: {clean_image.shape[0]}")
    logging.info(f"Labels: {label.shape}")

    fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x', fontsize=24)
    axs[1].set_title('noisy x', fontsize=24)
    axs[2].set_title('less noisy x', fontsize=24)

    axs[0].imshow(torchvision.utils.make_grid(clean_image)[0], cmap='Greys')
    axs[1].imshow(torchvision.utils.make_grid(corrupted_image)[0], cmap='Greys')
    axs[2].imshow(torchvision.utils.make_grid(less_corrupted_image)[0], cmap='Greys')
    plt.savefig(os.path.join(save_dir,'Corruption_pairs_sample.png'), bbox_inches='tight')
    # plt.show()
    plt.close()

    corruptor = AVAILABLE_CORRUPTORS[config.solver.type](
        config=config,
        transform=config.data.transform)

    # get_initial_corrupted_sample
    n_denoising_steps = config.solver.n_denoising_steps
    initial_corrupted_sample, clean_initial_sample, intermediate_corruption_samples = sampling.get_initial_corrupted_sample(
        trainloader, n_denoising_steps, corruptor)

    utils.save_gif(save_dir, intermediate_corruption_samples, "corruption_init.gif")
    utils.save_video(save_dir, intermediate_corruption_samples, filename="corruption_init.mp4")
    utils.save_png(save_dir, clean_initial_sample, "clean_init.png")

    fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
    axs[0].set_title('clean x', fontsize=24)
    axs[1].set_title('noisy x', fontsize=24)

    axs[0].imshow(torchvision.utils.make_grid(clean_initial_sample)[0], cmap='Greys')
    axs[1].imshow(torchvision.utils.make_grid(initial_corrupted_sample)[0], cmap='Greys')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fully_corrupted_sample.png'), bbox_inches='tight')
    # plt.show()
    plt.close()

if __name__ == '__main__':

    # config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True) # this does not work in python 3.12 as 'imp' module has been removed
    flags.DEFINE_string("config", None, "Path to the config file.")
    flags.DEFINE_boolean("demo", False, "Runs the script in demonstration mode.")
    flags.mark_flags_as_required(["config"])

    app.run(main)
