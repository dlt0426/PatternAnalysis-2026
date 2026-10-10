"""
Data loading and patient-level splitting for the ADNI AD vs NC dataset.

File names follow '<patientID>_<sliceIndex>.jpeg' (20 slices per patient).
All slices of one patient are kept in the same split to prevent leakage.
"""
import os
import glob
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
CLASSES = {"NC": 0, "AD": 1}
IMG_SIZE = 224


def patient_id(path):
    """'.../218391_78.jpeg' -> '218391'"""
    return os.path.basename(path).split("_")[0]


def collect(split_dir):
    """Return a list of (image_path, label) for one official split folder."""
    items = []
    for name, label in CLASSES.items():
        for p in sorted(glob.glob(os.path.join(split_dir, name, "*.jpeg"))):
            items.append((p, label))
    return items


def split_by_patient(items, val_frac=0.15, seed=42):
    """Stratified patient-level split of `items` into (train, val)."""
    by_patient = defaultdict(list)
    for p, y in items:
        by_patient[(patient_id(p), y)].append((p, y))

    rng = random.Random(seed)
    train, val = [], []
    for label in CLASSES.values():
        keys = sorted(k for k in by_patient if k[1] == label)
        rng.shuffle(keys)
        n_val = int(len(keys) * val_frac)
        for i, k in enumerate(keys):
            (val if i < n_val else train).extend(by_patient[k])
    return train, val


def get_splits(root=DATA_ROOT, val_frac=0.15, seed=42):
    """Official test set + patient-level train/val split of the official train set."""
    train_all = collect(os.path.join(root, "train"))
    test = collect(os.path.join(root, "test"))
    train, val = split_by_patient(train_all, val_frac, seed)
    return train, val, test


class ADNIDataset(Dataset):
    """Returns (image, label, patient_id) for each slice."""

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, y, patient_id(path)


def compute_mean_std(items, n=2000, seed=0):
    """Estimate pixel mean/std from a random subset of TRAINING images only
    (using val/test statistics would leak information)."""
    rng = random.Random(seed)
    subset = rng.sample(items, min(n, len(items)))
    to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
    total, total_sq, count = 0.0, 0.0, 0
    for p, _ in subset:
        x = to_tensor(Image.open(p).convert("L"))
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        count += x.numel()
    mean = total / count
    std = (total_sq / count - mean ** 2) ** 0.5
    return mean, std


def get_transforms(mean, std, train):
    """Augmentation is applied to the training set only; val/test are deterministic."""
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),                       # brains are roughly left-right symmetric
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize([mean], [std]),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([mean], [std]),
    ])


def get_loaders(batch_size=64, num_workers=4, root=DATA_ROOT, smoke=False):
    """Build train/val/test DataLoaders.

    smoke=True keeps only a small random subset of each split so the whole
    pipeline can be checked in a minute before launching a full run.
    """
    train, val, test = get_splits(root)
    if smoke:
        rng = random.Random(0)
        train = rng.sample(train, min(512, len(train)))
        val = rng.sample(val, min(256, len(val)))
        test = rng.sample(test, min(256, len(test)))

    mean, std = compute_mean_std(train, n=200 if smoke else 2000)

    def make(items, train_mode):
        return DataLoader(
            ADNIDataset(items, get_transforms(mean, std, train_mode)),
            batch_size=batch_size, shuffle=train_mode,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

    loaders = {"train": make(train, True), "val": make(val, False), "test": make(test, False)}
    return loaders, (mean, std)
