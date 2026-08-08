import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

import config


def _patient_id(path):
    """Extract patient/scan ID from filename by stripping _sliceN suffix."""
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'_s\d+$', '', name)


class ImageDataset(Dataset):
    def __init__(self, img_dir, augment=False):
        self.paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])

        self.base = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.aug = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.aug(img) if self.augment else self.base(img)


def get_splits():
    full = ImageDataset(config.NORMAL_DIR, augment=False)
    n = len(full)
    train_size = int(n * config.SPLITS[0])
    val_size = int(n * config.SPLITS[1])
    test_size = n - train_size - val_size

    train_set, val_set, test_set = random_split(
        full, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    train_set.dataset = ImageDataset(config.NORMAL_DIR, augment=True)
    return train_set, val_set, test_set


class MetaImageDataset(Dataset):
    """Returns (image, patient_id, filepath) for heatmap grouping."""

    def __init__(self, img_dir):
        self.paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(path))
        return img, _patient_id(path), path


def get_test_loaders():
    _, _, normal_test = get_splits()
    abnormal = ImageDataset(config.ABNORMAL_DIR, augment=False)

    normal_loader = DataLoader(normal_test, batch_size=config.BATCH_SIZE, shuffle=False)
    abnormal_loader = DataLoader(abnormal, batch_size=config.BATCH_SIZE, shuffle=False)
    return normal_loader, abnormal_loader


def get_test_loaders_all_normal():
    """Uses ALL normal images for evaluation, not just the 15% test split."""
    normal_all = ImageDataset(config.NORMAL_DIR, augment=False)
    abnormal = ImageDataset(config.ABNORMAL_DIR, augment=False)
    normal_loader = DataLoader(normal_all, batch_size=config.BATCH_SIZE, shuffle=False)
    abnormal_loader = DataLoader(abnormal, batch_size=config.BATCH_SIZE, shuffle=False)
    return normal_loader, abnormal_loader


def get_test_loaders_with_meta():
    """Returns loaders that yield (image, patient_id, filepath) tuples."""
    normal_meta = MetaImageDataset(config.NORMAL_DIR)
    abnormal_meta = MetaImageDataset(config.ABNORMAL_DIR)
    normal_loader = DataLoader(normal_meta, batch_size=config.BATCH_SIZE, shuffle=False)
    abnormal_loader = DataLoader(abnormal_meta, batch_size=config.BATCH_SIZE, shuffle=False)
    return normal_loader, abnormal_loader
