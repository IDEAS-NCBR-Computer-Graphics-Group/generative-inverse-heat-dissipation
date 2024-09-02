
# env installation

image:
c2-deeplearning-pytorch-1-13-cu113-v20240730-debian-11

## conda way
create and activate conda env

```
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

$ nvcc --version # check your version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113 # and install the right one

pip install -r gcp_requirements.txt
```

### athena cluster

module load CUDA/12.4.0
module load Python/3.10.4

source $SCRATCH/ihd-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

### datasets download

#### afhq
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


#### ffhq

Go to the website and "Create New API Token". This will download a kaggle.json

```
pip install kaggle
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d gibi13/flickr-faces-hq-dataset-ffhq     # ~80GB
kaggle datasets download -d potatohd404/ffhq-128-70k                # ~2GB
unzip flickr-faces-hq-dataset-ffhq.zip
```

url:
https://www.kaggle.com/datasets/potatohd404/ffhq-128-70k


# check disk usage

sudo df -h 
du -sh * | sort -hr | head -n10
ncdu $HOME