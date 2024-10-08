# Generative Modelling With Inverse Heat Dissipation
 
This repository is the official implementation of the methods in the publication:

* Severi Rissanen, Markus Heinonen, and Arno Solin (2023). **Generative Modelling With Inverse Heat Dissipation**. In *International Conference on Learning Representations (ICLR)*. [[arXiv]](https://arxiv.org/abs/2206.13397) [[project page]](https://aaltoml.github.io/generative-inverse-heat-dissipation)

## Arrangement of code

The "`configs`" folder contains the configuration details on different experiments and the "`data`" folder contains the data. MNIST and CIFAR-10 should run as-is with automatic torchvision data loading, but the other experiments require downloading the data to the corresponding `data/` folders. The "`model_code`" contains the U-Net definition and utilities for working with the proposed inverse heat dissipation model. "`scripts`" contains additional code, for i/o, data loading, loss calculation and sampling. "`runs`" is where the results get saved at.

## Used Python packages

generative-inverse-heat-dissipation
The file "requirements.txt" contains the Python packages necessary to run the code, and they can be installed by running

```pip install -r requirements.txt```

If you have issues with installing the `mpi4py` through pip, you can also install it using conda with `conda install -c conda-forge mpi4py`.

## Training

You can get started by running an MNIST training script with

```python train.py --config configs/mnist/small_mnist.py --workdir runs/mnist/small_mnist```

```python train.py --config configs/mnist/default_mnist_configs.py --workdir runs/mnist/default```

```python train.py --config configs/cifar10/default_cifar10_configs.py --workdir runs/cifar10/default```

```python train.py --config configs/afhq/default_afhq_configs.py --workdir runs/afhq/default```

```python train.py --config configs/afhq/img_size_128.py --workdir runs/afhq/img_size_128```

```python train.py --config configs/ffhq/default_ffhq_configs.py --workdir runs/ffhq/default```

```python train.py --config configs/ffhq/small_img_size_128.py --workdir runs/ffhq/small_img_size_128```


```python train_corrupted.py --config configs/ffhq/128_ffhq_lbm_ns_config.py --workdir runs/ffhq/128_ffhq_lbm_ns```

This creates a folder "`runs/mnist/default`", which contains the folder "`checkpoint-meta`", where the newest checkpoint is saved periodically. "`samples`" folder contains samples saved during training. You can change the frequency of checkpointing and sampling with the command line flags "`training.snapshot_freq_for_preemption=?`" and "`config.training.sampling_freq=?`". 

## Sampling

Once you have at least one checkpoint, you can do sampling with "`sample.py`", with different configurations:

### Random samples

Random samples:

```bash
python sample.py --config configs/mnist/default_mnist_configs.py --workdir runs/mnist/default --checkpoint 0 --batch_size=9

python sample.py --config configs/ffhq/img_size_128_maxblur128.py --workdir runs/ffhq/img_size_128_maxblur128 --checkpoint 0 --batch_size=9
```

### Share the initial state

Samples where the prior state u_K is fixed, but the sampling noise is different:

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --same_init
```

### Share the noise

Samples where the prior state u_K changes, but the sampling noises are shared (results in similar overall image characteristics, but different average colours if the maximum blur is large enough):

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --batch_size=9
                 --share_noise
 ```

### Interpolation

Produces an interpolation between two random points generated by the model.

```bash
python sample.py --config configs/mnist/default_mnist_configs.py
                 --workdir runs/mnist/default --checkpoint 0 --interpolate --num_points=20
```

## Evaluation

The script "`evaluation.py`" contains code for evaluating the model with FID-scores and NLL (ELBO) values. For example, if you have a trained cifar-10 model trained with `configs/cifar10/default_cifar10_configs.py` and the result is in the folder `runs/cifar10/default/checkpoints-meta`, you can run the following (checkpoint=0 refers to the last checkpoint, other checkpoints in `runs/cifar10/default/checkpoints` are numbered as 1,2,3,...):

### FID scores
This assumes that you have `clean-fid` installed. 

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py --workdir runs/cifar10/default --checkpoint 0 --dataset_name=cifar10 --experiment_name=experiment1 --param_name=default --mode=fid --delta=0.013 --dataset_name_cleanfid=cifar10 --dataset_split=train --batch_size=128 --num_gen=50000
```

### NLL values
The result contains a breakdown of the different terms in the NLL.

```bash
python evaluate.py --config configs/cifar10/default_cifar10_configs.py --workdir runs/cifar10/default --checkpoint 0 --dataset_name=cifar10 --experiment_name=experiment1 --param_name=default --mode=elbo --delta=0.013
```

### Result folder
The results will be saved in the folder `runs/cifar10/evaluation_results/experiment1/` in log files, where you can read them out. The idea in general is that `experiment_name` is an upper-level name for a suite of experiments that you might want to have (e.g., FID w.r.t. different delta), and `param_name` is the name of the calculated value within that experiment (e.g., "delta0.013" or "delta0.012"). 

## Citation

If you use the code in this repository for your research, please cite the paper as follows:

```bibtex
@inproceedings{rissanen2023generative,
  title={Generative modelling with inverse heat dissipation},
  author={Severi Rissanen and Markus Heinonen and Arno Solin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

This software is provided under the [MIT license](LICENSE).

