import os
from pathlib import Path
import logging
from scripts import losses
from scripts import sampling
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import numpy as np

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
#flags.DEFINE_string("initialization", "prior", "How to initialize sampling")


def main(argv):
    train(FLAGS.config, FLAGS.workdir)


def train(config, workdir):
    """Runs the training pipeline. 
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """

    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")

    # Create directory for saving intermediate samples
    sample_dir = os.path.join(workdir, "samples")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    # Create directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # # Initialize model
    # model = mutils.create_model(config)
    # optimizer = losses.get_optimizer(config, model.parameters())
    # ema = ExponentialMovingAverage(
    #     model.parameters(), decay=config.model.ema_rate)
    # state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    # model_evaluation_fn = mutils.get_model_fn(model, train=False)

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
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the forward process definition
    # scales = config.model.blur_schedule
    # heat_forward_module = None

    # Get the loss function
    train_step_fn = losses.get_step_lbm_fn(train=True, config=config, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_lbm_fn(train=False, config=config, optimize_fn=optimize_fn)

    # Building sampling functions
    # delta = config.model.sigma*1.25
    # initial_sample, _ = sampling.get_initial_sample(
    #     config, heat_forward_module, delta)
    
    # TODO: draw a sample by lbm-destroying some rand images?
    from numerical_solvers.data_holders.LBM_NS_Corruptor import LBM_NS_Corruptor
    from torchvision import transforms
    from configs.mnist.lbm_ns_turb_config import get_lbm_ns_config, LBMConfig
    # lbm_corruptor = LBM_NS_Corruptor() 
    solver_config = get_lbm_ns_config()
    lbm_ns_Corruptor = LBM_NS_Corruptor(
        solver_config,                                
        transform=transforms.Compose([transforms.ToTensor()]))
    
    def get_initial_lbm_sample(solver_config: LBMConfig, solver: LBM_NS_Corruptor, batch_size=None):
        """Take a draw from the prior p(u_K)"""
        trainloader, _ = datasets.get_dataset(config,
                                            uniform_dequantization=config.data.uniform_dequantization,
                                            train_batch_size=batch_size)

        # initial_sample = next(iter(trainloader))[0].to('cpu')
        initial_sample, _ = datasets.prepare_batch(iter(trainloader), 'cpu')
        corrupted_sample = torch.empty_like(initial_sample)
        # vec_corruption_amount = torch.randint(
        #     low=solver_config.solver.min_lbm_steps, 
        #     high=solver_config.solver.max_lbm_steps, 
        #     size=initial_sample.shape[0], device='cpu')
        
        for index in range(initial_sample.shape[0]):
            # corruption_amount = solver_config.solver.max_lbm_steps #TODO we shall start from completely destroyed images
            corruption_amount = np.random.randint(solver_config.solver.min_lbm_steps, solver_config.solver.max_lbm_steps)
            tmp, _ = solver._corrupt(initial_sample[index], 
                                     corruption_amount #vec_corruption_amount[index]
                                     )
            
            corrupted_sample[index] = tmp
        return corrupted_sample
    
    initial_sample = get_initial_lbm_sample(solver_config, lbm_ns_Corruptor)
    
    sampling_fn = sampling.get_sampling_fn_inverse_lbm_ns(
        solver_config.solver.max_lbm_steps, # vec_corruption_amount, 
        initial_sample, 
        intermediate_sample_indices=list(range(solver_config.solver.max_lbm_steps)),
        delta=config.model.sigma*1.25, device=config.device)

    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))

    # For analyzing the mean values of losses over many batches, for each scale separately
    # pooled_losses = torch.zeros(len(scales))

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        try:
            # batch = next(train_iter)[0].to(config.device).float()
            # _, batch = next(train_iter).to(config.device).float() # not that easy if batch has mutltiple elements
            # x, (y, pre_y, corruption_amount, labels) = next(train_iter)
            _, batch = datasets.prepare_batch(train_iter, config.device)
            
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            # batch = next(train_iter)[0].to(config.device).float()
            _, batch = datasets.prepare_batch(train_iter, config.device)
        loss, _, _ = train_step_fn(state, batch)

        writer.add_scalar("training_loss", loss.item(), step)

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info("Saving temporary checkpoint")
            utils.save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            logging.info("Starting evaluation")
            # Use 25 batches for test-set evaluation, arbitrary choice
            N_evals = 25
            for i in range(N_evals):
                try:
                    # eval_batch = next(eval_iter)[0].to(config.device).float()
                    _, eval_batch = datasets.prepare_batch(eval_iter, config.device)
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    _, eval_batch = datasets.prepare_batch(eval_iter, config.device)
                eval_loss, _, _ = eval_step_fn(state, eval_batch)
                eval_loss = eval_loss.detach()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info("Saving a checkpoint")
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)

        # Generate samples periodically
        if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps:
            logging.info("Sampling...")
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)
            ema.restore(model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "final.np")
            utils.save_png(this_sample_dir, sample, "final.png")
            utils.save_png_norm(this_sample_dir, sample, "final_norm.png") # TODO: make it consisten with the original pipeline
            
            if initial_sample != None:
                utils.save_png(this_sample_dir, initial_sample, "init.png")
                utils.save_png_norm(this_sample_dir, initial_sample, "init_norm.png")
                
            utils.save_gif(this_sample_dir, intermediate_samples)
            utils.save_video(this_sample_dir, intermediate_samples)

if __name__ == "__main__":
    app.run(main)