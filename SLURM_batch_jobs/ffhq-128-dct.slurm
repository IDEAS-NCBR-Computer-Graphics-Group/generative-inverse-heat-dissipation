#!/bin/bash
#SBATCH --job-name=ffhq-128
#SBATCH --time=04:00:00
#SBATCH --account=plgclb2024-gpu-a100
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2



# module load GCC/12.3.0
# module load GCCcore/12.3.0
# module load CUDA/12.4.0
# module load gompi/2023a


module load Python/3.10.4
echo "Job started on $(date) at $HOSTNAME"

source $SCRATCH/py-ihd-env/bin/activate
cd $SCRATCH/generative-inverse-heat-dissipation
# mpiexec 

CMD="python train.py --config configs/ffhq/small_img_size_128.py --workdir runs/ffhq/128_ffhq_dct"
echo "Executing CMD:"
echo $CMD

eval $CMD

echo "Job finished on $(date)"
