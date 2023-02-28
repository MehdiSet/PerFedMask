# PerFedMask: Personalized Federated Learning with Optimized Masking Vectors

This is the official Pytorch implementation of our paper [PerFedMask: Personalized Federated Learning with Optimized Masking Vectors](https://openreview.net/pdf?id=hxEIgUXLFF) accepted in ICLR 2023.

## Installation

First check the requirements as follows:\
python=3.7\
numpy=1.17.0\
pytorch=1.12.1\
cudatoolkit = 11.3.1\
wandb=0.12.19\
torchvision=0.13.1\
cvxpy=1.1.11\
mosek=9.2.40

Then clone the repository as follows:
```shell
git clone https://github.com/MehdiSet/PerFedMask.git
```

## Dataset

We conduct our experiments on CIFAR-10, CIFAR-100, and DomainNet datasets using ResNet (PreResNet18), MobileNet , and AlexNet, respectively. Please download the datasets and place them under `data/` directory.


## Citation

If you find our paper and code useful, please cite our paper as follows:
```bibtex
@inproceedings{setayesh2023perfedmask,
  title={PerFedMask: {Personalized} Federated Learning with Optimized Masking Vectors},
  author={Setayesh, Mehdi and Li, Xiaoxiao and W.S. Wong, Vincent},
  booktitle={Proc. of International Conference on Learning Representations (ICLR)},
  address={Kigali, Rwanda},
  month={May},
  year={2023}
}
```

## Contact

Please feel free to contact us if you have any questions:
- Mehdi Setayesh: setayeshm@ece.ubc.ca

## Acknowledgements
This codebase was adapted from https://github.com/illidanlab/SplitMix and https://github.com/jhoon-oh/FedBABU.

