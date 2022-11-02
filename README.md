# Sparse-Sharpness-Aware-Minimization

This is the official implementation of NeurIPS 2022 ["Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach"](https://arxiv.org/abs/2210.05177#) (in Pytorch).  

<!-- ## Introduction -->

## Installation
- Clone this repo
```bash
git clone git@github.com:Mi-Peng/Sparse-Sharpness-Aware-Minimization.git
cd Sparse-Sharpness-Aware-Minimization
```

- Create a virtual environment (e.g. Anaconda3)
```bash
conda create -n ssam python=3.8 -y
conda activate ssam
```
- Install the necessary packages

Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/locally/).
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Install wandb(optional)
```bash
pip install wandb
```

- Dataset preparation

We use CIFAR10, CIFAR100 and ImageNet in this repo.

For the CIFAR dataset, you don't need to do anything, pytorch will do the trivia about downloading.

For ImageNet dataset, we use standard ImageNet dataset, which could be found in  http://image-net.org/. Your ImageNet file structure should look like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```
## Config Introduction
In `configs/default_cfg.py`, we list the common args：
- `--dataset`. The dataset you used, choice: `CIFAR10_base`, `CIFAR10_cutout`, `CIFAR100_base`, `CIFAR100_cutout` and `ImageNet_base`.
- `--model`. The model you used, choice: `resnet18`, `vgg11_bn`, `wideresnet28x10`...
- `--opt`. The optimizer you used. `sgd` for SGD, `sam-sgd` for SAM (within SGD), `ssamf-sgd` for Fisher-SparseSAM (within SGD) and so on.
- `--sparsity`. The proportion of weight that does not calculate perturbation (Only works for ssamd and ssamf).
- `--update_freq`. How many epochs to update the mask (Only works for ssamd and ssamf).
- `--num_samples`. The number of samples to calculate Fisher Information (Only works for ssamf).
- `--drop_rate`. (Only works for ssamd). 
- `--drop_strategy`. choice: `weight`, `gradient`, `random` (Only works for ssamd). 
- `--growth_strategy`. choice:  `weight`, `gradient`, `random `(Only works for ssamd).
- `--wandb`. action="store_true", if you want to record the information on wandb.

## Training
Training Resnet18 on CIFAR10 with SGD as baseline.
```bash
python train.py --model resnet18 --dataset CIFAR10_cutout --datadir [Path2Data] --opt sgd --lr 0.05 --weight_decay 5e-4 --seed [Maybe3107?] --wandb
```

Take ResNet18 on CIFAR10 training with Fisher-SparseSAM as an example.
```bash
python train.py --model resnet18 --dataset CIFAR10_cutout --datadir [Path2Data] --opt ssamf-sgd --weight_decay 1e-3 --sparsity 0.5 --num_samples 16 --update_freq 1 --seed [Maybe3107?] --wandb
```

## Citation
```bash
@article{mi2022make,
  title={Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach},
  author={Mi, Peng and Shen, Li and Ren, Tianhe and Zhou, Yiyi and Sun, Xiaoshuai and Ji, Rongrong and Tao, Dacheng},
  journal={arXiv preprint arXiv:2210.05177},
  year={2022}
}
```