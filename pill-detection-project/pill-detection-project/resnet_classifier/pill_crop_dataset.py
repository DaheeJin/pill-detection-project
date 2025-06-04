
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import torch
import numpy as np

def setup_dataset_with_weights(data_dir, transform, val_ratio=0.1, batch_size=32, seed=42, device='cpu'):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = np.array([label for _, label in dataset])
    indices = np.arange(len(targets))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, targets))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_labels = [targets[i] for i in train_idx]
    class_counts = Counter(train_labels)
    num_classes = len(dataset.classes)
    class_sample_counts = [class_counts.get(i, 0) for i in range(num_classes)]

    class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    return dataset.classes, train_loader, val_loader, class_weights



def setup_bright_blur_dataloader(data_dir, transform, val_ratio=0.1, batch_size=32, device='cpu', seed=42):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = np.array([label for _, label in dataset])
    indices = np.arange(len(targets))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, targets))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_labels = [targets[i] for i in train_idx]
    class_counts = Counter(train_labels)
    num_classes = len(dataset.classes)
    class_sample_counts = [class_counts.get(i, 0) for i in range(num_classes)]

    class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    return dataset.classes, train_loader, val_loader, class_weights

