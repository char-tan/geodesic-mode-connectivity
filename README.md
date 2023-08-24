# Geodesic Mode Connectivity

This repository is the official implementation of [Geodesic Mode Connectivity](https://openreview.net/forum?id=cFtt9fU7YB6). 

![figure_1](/figs/figure_1.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Geodesic Optimisation

To run geodesic optimisation on pre-trained models simply run:

```train
python main.py
```

>The above command will default to the same configuration as used in the TinyPaper, using the same pre-trained models.

## Training Base Models

To run geodesic optimisation on newly trained models run:

```eval
python main.py --base_training
```

>Several hyperparameters can be passed into the base training, please see main.py for further details.

## Pre-trained Models

Two independently trained ResNet20 (width multiplier = 4) models are provided:

- model_files/model_a.pt
- model_files/model_b.pt

In addtion, a permuted version of model_b is provided, wherein weight matching [1] was employed with respect to model_a:

- model_files/model_b_p.pt

## Results

![figure_2](/figs/figure_2.png)

## References

[1] S. K. Ainsworth, J. Hayase, and S. Srinivasa, ‘Git Re-Basin: Merging Models modulo Permutation Symmetries’, arXiv [cs.LG]. 2023.

## Acknowledgements

The weight matching and permutation code is a minor adapation of [git-re-basin-pytorch](https://github.com/themrzmaster/git-re-basin-pytorch/tree/main).

The ResNet class is a minor adapation of [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
