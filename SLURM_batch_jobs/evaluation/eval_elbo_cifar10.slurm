#!/bin/bash
#SBATCH --job-name=eval-elbo-cifar10
#SBATCH --time=04:00:00
#SBATCH --account=plgclb2024-gpu-a100
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module load Python/3.10.4

echo $(date)
source $SCRATCH/py-ihd-env/bin/activate

cd $SCRATCH/generative-inverse-heat-dissipation
python evaluate.py --config configs/cifar10/default_cifar10_configs.py --workdir runs/cifar10/default --checkpoint 0 --dataset_name=cifar10 --experiment_name=experiment1a --param_name=default --mode=elbo --delta=0.013
