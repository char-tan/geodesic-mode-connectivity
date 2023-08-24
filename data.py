from typing import Tuple

import torch
import torchvision

CIFAR10_MEAN = torch.tensor(
    [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
)
CIFAR10_STD = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])


def get_dataloaders(
    crop_size: int, padding: int, batch_size: int
) -> Tuple[torch.utils.data.DataLoader]:

    """constructs and returns dataloaders for train and test"""

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(
                crop_size, padding=padding, padding_mode="reflect"
            ),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
