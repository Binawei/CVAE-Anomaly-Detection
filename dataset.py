import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config

class ImageDataSet(Dataset):
    def __init__(self, img_dir, augment=False):
        self.paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]

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


def get_spits():
    full_dataset = ImageDataSet(config.NORMAL_DIR, augment=False)
    n = len(full_dataset)
    train_size = int(n * config.SPLITS[0])
    val_size = int(n * config.SPLITS[1])
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(config.SEED))
    train_dataset.dataset = ImageDataSet(config.NORMAL_DIR, augment=True)
    return train_dataset, val_dataset, test_dataset


def get_test_loaders():
    _, _, normal_test = get_spits()
    abnormal = ImageDataSet(config.ABNORMAL_DIR, augment=False)

    normal_loader = DataLoader(normal_test, batch_size=config.BATCH_SIZE, shuffle=False)
    abnormal_loader = DataLoader(abnormal, batch_size=config.BATCH_SIZE, shuffle=False)
    return normal_loader, abnormal_loader