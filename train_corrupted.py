import os
import shutil
from pathlib import Path
import logging
from scripts import losses
from scripts import sampling
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
import numpy as np
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
import wandb

from numerical_solvers.data_holders.CorruptedDatasetCreator import AVAILABLE_CORRUPTORS
from scripts.git_utils import get_git_branch, get_git_revision_hash, get_git_revision_short_hash
from scripts.utils import load_config_from_path, setup_logging

FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file("config", None, "NN Training configuration.", lock_config=True) # removed in python 3.12 # this return a parsed object - ConfigDict
flags.DEFINE_string("config", None, "Path to the config file.")
flags.mark_flags_as_required(["config"])


def main(argv):
    # Example
    # python train_corrupted.py --config=configs/ffhq/res_128/ffhq_128_lbm_ns_config_lin_visc.py
    train(FLAGS.config)

def train(config_path):
    """Runs the training pipeline. 
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Initial logging setup
    logging.basicConfig(level=logging.DEBUG)

    # Load config
    config = load_config_from_path(config_path)

    case_name = f"{config.data.processed_filename}_{config.stamp.fwd_solver_hash}_{config.stamp.model_optim_hash}"
    
    wandb.init(
        project='fluid-diffusion',
        config=config,
        name= case_name
    )

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Setup working directory path 
    workdir = os.path.join(f'runs/corrupted_{config.data.dataset}',case_name)

    # copy config to know what has been run
    Path(workdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, workdir) 
    print(os.path.join(*config_path.split(os.sep)[:-1], f'default_lbm_{config.data.dataset.lower()}_config.py'))
    shutil.copy(os.path.join(*config_path.split(os.sep)[:-1], f'default_lbm_{config.data.dataset.lower()}_config.py'), workdir)

    # Setup logging once the workdir is known
    setup_logging(workdir)

    logging.info(f"Code version\t branch: {get_git_branch()} \t commit hash: {get_git_revision_hash()}")
    logging.info(f"Execution flags: --config: {config_path}")
    logging.info(f"Run directory: {workdir}")
    
    if config.device == torch.device('cpu'):
        logging.warning("RUNNING ON CPU")

    # Create directory for saving intermediate samples
    sample_dir = os.path.join(workdir, "samples")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    # Create directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model
    model = mutils.create_model(config)
    logging.info(f"No of network parameters: {sum([p.numel() for p in model.parameters()])}")
    # logging.info(f"NN architecture:\n{model}")
    
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    model_evaluation_fn = mutils.get_model_fn(model, train=False)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(
        workdir, "checkpoints-meta", "checkpoint.pth")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(checkpoint_meta_dir)).mkdir(
        parents=True, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    trainloader, testloader = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization)
    datadir = os.path.join(f'data/corrupted_{config.data.dataset}',f'{config.data.processed_filename}_{config.stamp.fwd_solver_hash}')
    shutil.copy(config_path, datadir)
    shutil.copy(os.path.join(*config_path.split(os.sep)[:-1], f'default_lbm_{config.data.dataset.lower()}_config.py'), datadir)
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the loss function
    train_step_fn = losses.get_step_lbm_fn(train=True, config=config, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_lbm_fn(train=False, config=config, optimize_fn=optimize_fn)

    # Building sampling functions
    # Get the forward process definition

    corruptor=AVAILABLE_CORRUPTORS[config.solver.type](
        config=config,
        transform=config.data.transform
    )

    # draw a sample by destroying some rand images 
    n_denoising_steps = config.solver.n_denoising_steps
    initial_corrupted_sample, clean_initial_sample, intermediate_corruption_samples = sampling.get_initial_corrupted_sample(
        trainloader, n_denoising_steps, corruptor)
    
    logging.info("Saving forward corruption process.")
    utils.save_gif(workdir, intermediate_corruption_samples, "corruption_init.gif")
    video_mpv4_filename, video_x264_filename = "corruption_init.mp4", "corruption_init_x264.mp4"
    video_mpv4_path, video_x264_path = os.path.join(workdir, video_mpv4_filename), os.path.join(workdir, video_x264_filename)
    utils.save_video(workdir, intermediate_corruption_samples, filename=video_mpv4_filename)
    os.system(f'ffmpeg -i {video_mpv4_path} -vcodec libx264 -f mp4 {video_x264_path}')
    utils.save_png(workdir, clean_initial_sample, "clean_init.png")
    wandb.log({
        "clean_init": wandb.Image(os.path.join(workdir, "clean_init.png")),
        "corruption_init": wandb.Video(video_x264_path)
        })


    sampling_fn = sampling.get_sampling_fn_inverse_lbm_ns(
        n_denoising_steps = n_denoising_steps,
        initial_sample = initial_corrupted_sample, 
        intermediate_sample_indices=list(range(n_denoising_steps+1)), # assuming n_denoising_steps=3, then intermediate_sample_indices = [0, 1, 2, 3]
        delta=config.model.sigma*1.25, 
        device=config.device)

    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))

    # For analyzing the mean values of losses over many batches, for each scale separately
    # pooled_losses = torch.zeros(len(scales))

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        try:
            _, batch = datasets.prepare_batch(train_iter, config.device)
            
        except StopIteration:  # Start new epoch if run out of data
            logging.info(f"New epoch at step={step}.")
            train_iter = iter(trainloader)
            _, batch = datasets.prepare_batch(train_iter, config.device)
        loss, _, _ = train_step_fn(state, batch)

        writer.add_scalar("training_loss", loss.item(), step)
        wandb.log({'training_loss': loss.item()})

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info(f"Saving temporary checkpoint at step={step}.")
            utils.save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            logging.info(f"Starting evaluation on test dataset at step={step}.")
            # Use training.n_evals of batches for test-set evaluation, arbitrary choice
            for i in range(config.training.n_evals):
                try:
                    # eval_batch = next(eval_iter)[0].to(config.device).float()
                    _, eval_batch = datasets.prepare_batch(eval_iter, config.device)
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    _, eval_batch = datasets.prepare_batch(eval_iter, config.device)
                eval_loss, _, _ = eval_step_fn(state, eval_batch)
                eval_loss = eval_loss.detach()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            wandb.log({'eval_loss': eval_loss.item()})

        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info(f"Saving a checkpoint at step={step}")
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)

        # Generate samples periodically
        if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps:
            logging.info(f"Sampling at step={step}...")
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)
            ema.restore(model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "final.np")
            utils.save_png(this_sample_dir, sample, "final.png")

            if initial_corrupted_sample != None:
                utils.save_png(this_sample_dir, initial_corrupted_sample, "init.png")
                wandb.log({'init': wandb.Image(os.path.join(this_sample_dir, 'init.png')),
                           'final': wandb.Image(os.path.join(this_sample_dir, 'final.png'))})
            else:
                wandb.log({'final': wandb.Image(os.path.join(this_sample_dir, 'final.png'))})


            utils.save_gif(this_sample_dir, intermediate_samples)
            video_mpv4_filename, video_x264_filename = "process.mp4", "process_x264.mp4"
            video_mpv4_path, video_x264_path = os.path.join(this_sample_dir, video_mpv4_filename), os.path.join(this_sample_dir, video_x264_filename)
            utils.save_video(this_sample_dir, intermediate_samples, filename=video_mpv4_filename)
            os.system(f'ffmpeg -i {video_mpv4_path} -vcodec libx264 -f mp4 {video_x264_path}')
            wandb.log({"process": wandb.Video(video_x264_path)})



    logging.info("Done.")
     
if __name__ == "__main__":
    app.run(main)
