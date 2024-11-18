import torch
import numpy as np
import taichi as ti
import cv2
from absl import app
from absl import flags

from numerical_solvers.corruptors.CorruptedDatasetCreator import AVAILABLE_CORRUPTORS
from scripts.utils import load_config_from_path
from scripts import datasets
from torchvision.utils import make_grid

FLAGS = flags.FLAGS

def main(_):
    ti.init(arch=ti.gpu) if torch.cuda.is_available() else ti.init(arch=ti.cpu)
    
    config = load_config_from_path(FLAGS.config)
    if getattr(config, 'solver', None): 
        solver = config.solver
        config.solver = None
        
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainloader, _ = datasets.get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization,
        train_batch_size=16
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

if __name__ == '__main__':    
    flags.DEFINE_string("config", None, "Path to the config file.")
    flags.mark_flags_as_required(["config"])

    app.run(main)
