import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

__all__ = ["cifar10_train_dataset", "cifar10_val_dataset", "cifar10_test_dataset"]


indice = torch.randperm(50000)
train_indice = indice[: int(len(indice) * 0.8)]
val_indice = indice[int(len(indice) * 0.8) :]
mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.4914, 0.4822, 0.4465),
}
std = {
    "cifar10": (0.2470, 0.2435, 0.2616),
    "cifar100": (0.2023, 0.1994, 0.2010),
}


class Rescale:
    def __call__(self, img: torch.Tensor):
        return img * 255.0


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Rescale()
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
            Rescale()
        ]
    ),
}


def cifar10_train_dataset(root: str = "/your_path/cifar10"):
    return Subset(
        CIFAR10(root, train=True, transform=data_transforms["train"], download=True),
        indices=train_indice,
    )


def cifar10_val_dataset(root: str = "/your_path/cifar10"):
    return Subset(
        CIFAR10(root, train=True, transform=data_transforms["test"], download=True),
        indices=val_indice,
    )


def cifar10_test_dataset(root: str = "/your_path/cifar10"):
    return CIFAR10(root, train=False, transform=data_transforms["test"], download=True)
