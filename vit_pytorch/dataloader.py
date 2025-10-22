import os
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_dataloaders(train_dir: Optional[str] = None, test_dir: Optional[str] = None, batch_size: int = 32, image_size: int = 224, num_workers: int = 4) -> Tuple[Optional[DataLoader], Optional[DataLoader], int]:
    """Create dataloaders for training and/or testing.

    Both `train_dir` and `test_dir` are optional; at least one must be provided.
    Returns (train_loader_or_None, test_loader_or_None, num_classes)
    num_classes is inferred from the provided dataset (train first, else test).
    """
    if train_dir is None and test_dir is None:
        raise ValueError('Either train_dir or test_dir must be provided')

    # basic transforms
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_loader = None
    test_loader = None
    num_classes = 0

    if train_dir is not None:
        train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
        num_classes = len(train_ds.classes)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if test_dir is not None:
        test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
        if num_classes == 0:
            num_classes = len(test_ds.classes)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_classes
