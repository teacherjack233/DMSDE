import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset



def get_transform(dataset_name, img_size=32):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    if dataset_name == "cifar10":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_class(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return datasets.MNIST
    if dataset_name == "cifar10":
        return datasets.CIFAR10
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_task_labels(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in ["mnist", "cifar10"]:
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def _sample_indices(length, fraction):
    if fraction < 1.0:
        return np.random.choice(length, int(length * fraction), replace=False)
    return np.arange(length)


def get_dataset_tasks(
    dataset_name,
    num_tasks=5,
    fraction=0.5,
    batch_size=None,
    train_batch_size=64,
    test_batch_size=10,
    num_workers=0,
    seed=42,
):
    assert 0 < fraction <= 1.0
    if batch_size is not None:
        train_batch_size = batch_size
        test_batch_size = batch_size
    del train_batch_size
    _set_seed(seed)

    dataset_name = dataset_name.lower()
    dataset_cls = get_dataset_class(dataset_name)
    transform = get_transform(dataset_name)
    full_train = dataset_cls(root="./data", train=True, download=True, transform=transform)
    full_test = dataset_cls(root="./data", train=False, download=True, transform=transform)

    sampled_train_idx = _sample_indices(len(full_train), fraction)
    sampled_test_idx = _sample_indices(len(full_test), fraction)
    task_labels = get_task_labels(dataset_name)
    assert num_tasks <= len(task_labels)

    train_targets_np = np.array(full_train.targets)
    test_targets_np = np.array(full_test.targets)
    task_train_datasets = []
    task_test_loaders = []

    for task_id in range(num_tasks):
        labels = task_labels[task_id]
        train_idx = [idx for idx in sampled_train_idx if train_targets_np[idx] in labels]
        test_idx = [idx for idx in sampled_test_idx if test_targets_np[idx] in labels]
        if not train_idx or not test_idx:
            raise RuntimeError(
                f"Task {task_id} (labels={labels}) has an empty split. "
                "Check fraction or dataset."
            )

        train_subset = Subset(full_train, train_idx)
        test_subset = Subset(full_test, test_idx)
        task_train_datasets.append(train_subset)
        task_test_loaders.append(
            DataLoader(test_subset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
        )

    return task_train_datasets, task_test_loaders


def get_task_loader(task_data, batch_size):
    return DataLoader(task_data, batch_size=batch_size, shuffle=True, drop_last=True)


def get_test_loader(dataset_name, batch_size=64, task_idx=0, img_size=32):
    dataset_name = dataset_name.lower()
    transform = get_transform(dataset_name, img_size)
    task_labels = get_task_labels(dataset_name)
    labels_to_include = task_labels[task_idx]

    dataset_cls = get_dataset_class(dataset_name)
    dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)

    targets = dataset.targets
    mask = torch.tensor([label in labels_to_include for label in targets])
    if dataset_name == "cifar10":
        dataset.data = dataset.data[mask.numpy()]
        dataset.targets = torch.tensor(targets)[mask]
    else:
        dataset.data = dataset.data[mask]
        dataset.targets = targets[mask]

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
