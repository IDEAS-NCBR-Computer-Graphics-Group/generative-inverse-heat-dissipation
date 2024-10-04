
# env installation

image:
c2-deeplearning-pytorch-1-13-cu113-v20240730-debian-11

## conda way

create and activate conda env

```.sh
conda create --name ihd-env --clone base
conda activate ihd-env
pip install -r gcp_requirements.txt


$conda env remove --name ihd-env # if things go wrong
```

## python way

first time...`sudo apt-get install python3-dev`

then

```.sh

/usr/bin/python3.9 -m venv py-ihd-env
source ./py-ihd-env/bin/activate

pip install -r ihd_requirements.txt
pip install -r taichi_requirements.txt



python -m ipykernel install --user --name=py-ihd-env --display-name "py-ihd-env"



#if you face issues with cuda version
$ nvcc --version # check your version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113 # and install the right one



```

### athena cluster

```.sh
srun -N1 -n8 --account=plgclb2024-gpu-a100 --partition=plgrid-gpu-a100 --gres=gpu:1 --time=08:00:00 --pty /bin/bash -l
srun -N1 -n8 --account=plgclb2024-gpu-a100 --partition=plgrid-gpu-a100 --gres=gpu:1 --time=01:00:00 --pty /bin/bash -l

module load GCC/12.3.0
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load gompi/2023a
module load Python/3.10.4


python -m venv py-ihd-env
source $SCRATCH/py-ihd-env/bin/activate
$SCRATCH/py-ihd-env/bin/python -m pip install --upgrade pip


## skip the mpi4py part , we dont use it
#export MPICC=mpicc
#export MPI4PY_CC=mpicc
#pip install --no-binary=mpi4py mpi4py

#pip install mpi4py torch torchvision #--index-url https://download.pytorch.org/whl/cu124 # 116
pip install -r generative-inverse-heat-dissipation/athena_requirements.txt
pip install -r pi-inr/requirements.txt


cd /net/tscratch/people/plgmuaddieb/generative-inverse-heat-dissipation
python train.py --config configs/mnist/default_mnist_configs.py --workdir runs/mnist/default
```

#### some useless experiments

Generally mpi4py doesnt seem to work properly the pytorch code.
Just commented it away.

```.bash
#module load libtirpc/1.3.2
#module load OpenMPI/4.1.2-CUDA-11.6.0
```

run interactive job

```.bash
srun -N1 -n8 --account=plgclb2024-gpu-a100 --partition=plgrid-gpu-a100 --gres=gpu:1 --time=08:00:00 --pty /bin/bash -l
```

```.bash
export CPPFLAGS="-I/path/to/mpi/include"
export LDFLAGS="-L/path/to/mpi/lib"
pip install --no-binary=mpi4py mpi4py


#!/bin/bash

#Load the MPI module that supports SLURM (adjust this according to your environment)
module load openmpi  # Replace with `mpich` or your specific MPI module if needed

#Determine the include and library paths using `mpicc` and `mpicxx`
MPI_INCLUDE_PATH=$(mpicc --showme:compile | grep -oE '\-I[^ ]+' | sed 's/-I//')
MPI_LIB_PATH=$(mpicc --showme:link | grep -oE '\-L[^ ]+' | sed 's/-L//')

#Print the paths for verification
echo "MPI Include Path: $MPI_INCLUDE_PATH"
echo "MPI Library Path: $MPI_LIB_PATH"

#Export the environment variables for compilation 
export CPPFLAGS="-I/net/software/v1/software/OpenMPI/4.1.5-GCC-12.3.0/include"
export LDFLAGS="-L/net/software/v1/software/OpenMPI/4.1.5-GCC-12.3.0/lib -L/net/software/v1/software/hwloc/2.9.1-GCCcore-12.3.0/lib -L/net/software/v1/software/libevent/2.1.12-GCCcore-12.3.0/lib -Wl,-rpath -Wl,/net/software/v1/software/OpenMPI/4.1.5-GCC-12.3.0/lib -Wl,-rpath -Wl,/net/software/v1/software/hwloc/2.9.1-GCCcore-12.3.0/lib -Wl,-rpath -Wl,/net/software/v1/software/libevent/2.1.12-GCCcore-12.3.0/lib -Wl,--enable-new-dtags -lmpi"

#Export variables for mpi4py build process
export MPICC=mpicc
export MPI4PY_CC=mpicc

#Install mpi4py using pip with source compilation (no binary)
pip install --no-binary=mpi4py mpi4py

#Verification step: check if mpi4py was installed successfully
mpiexec python -c "from mpi4py import MPI; print('mpi4py is installed correctly.')" || echo "Installation failed."

echo "mpi4py installation completed."
```

mpicc -o mpi_hello_world mpi_hello_world.c
mpiexec ./mpi_hello_world

## datasets download

### afhq

https://github.com/clovaai/stargan-v2

in stargan-v2/download.sh

```.sh
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=./data/afhq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE
```

### ffhq

Go to the website and "Create New API Token". This will download a kaggle.json

```.sh
pip install kaggle
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d gibi13/flickr-faces-hq-dataset-ffhq     # ~80GB
unzip flickr-faces-hq-dataset-ffhq.zip

kaggle datasets download -d potatohd404/ffhq-128-70k                # ~2GB
mkdir -p ffhq-128-70k
unzip ffhq-128-70k.zip -d ffhq-128-70k
```

url:
<https://www.kaggle.com/datasets/potatohd404/ffhq-128-70k>

### mount disk on gcp

lsblk  
sudo fdisk -l

sudo fdisk -l /dev/nvme0n2 # Check the Disk for Partitions or Filesystem:
sudo mkfs.ext4 /dev/nvme0n2 # Create a Filesystem on the Disk (if needed):
sudo mount /dev/nvme0n2 /mnt/datadisk/

### qpa error

QObject::moveToThread: Current thread (0x7113010) is not the object's thread (0x99b7750).
Cannot move to target thread (0x7113010)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/computergraphics/anaconda3/envs/ihd-env/lib/python3.11/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, minimal, minimalegl, offscreen, vnc, webgl.

<https://stackoverflow.com/questions/71088095/opencv-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it-was-fou>

it was enought to reinstall:

```.sh
pip uninstall PyQt5
pip uninstall opencv-python
pip install opencv-python
```

maybe `pip install opencv-python-headless`

# check disk usage

```.sh
sudo df -h
du -sh * | sort -hr | head -n10
ncdu $HOME
hpc-fs # athena util
```

# runs

```.sh
conda activate ihd-env
export PYTHONPATH=$(pwd) # may be helpfull
python numerical_solvers/data_holders/CorruptedDatasetCreator.py
python numerical_solvers/runners/taichi_lbm_NS_picture_diffuser.py

python train.py --config configs/mnist/small_mnist.py --workdir runs/mnist/small_mnist

python train_corrupted.py --config configs/mnist/small_mnist_lbm_ns_config.py  --workdir runs/mnist/small_lbm_mnist  

python train_corrupted.py --config configs/mnist/small_mnist_lbm_ns_turb_config.py --workdir runs/mnist/small_lbm_turb_mnist

python train_corrupted.py --config configs/mnist/small_mnist_gaussian_blurring_config.py --workdir runs/mnist/small_gaussian_blurr_mnist
```
