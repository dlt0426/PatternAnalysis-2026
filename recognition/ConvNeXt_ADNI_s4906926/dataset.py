"""
Data loading and patient-level splitting for the ADNI AD vs NC dataset.

File names follow '<patientID>_<sliceIndex>.jpeg' (20 slices per patient).
All slices of one patient are kept in the same split to prevent leakage.
"""
import os
import glob
import json
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
# Scan-ID -> subject-group mapping produced by `python audit_split.py --make-groups`.
# Scan IDs whose images are near-identical are treated as the same subject.
GROUPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "train_groups.json")
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


def load_groups(path=GROUPS_FILE):
    """Return {scan_id: group_id}, or None if no grouping file exists."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["groups"]


def split_by_patient(items, val_frac=0.15, seed=42, groups=None):
    """Stratified split of `items` into (train, val) at the subject level.

    Without `groups`, each scan ID is its own unit. With `groups`, all scan IDs
    mapped to the same group (repeat scans of one subject) move together, so a
    subject can never appear in both train and val.
    """
    groups = groups or {}
    by_group = defaultdict(list)
    for p, y in items:
        pid = patient_id(p)
        by_group[groups.get(pid, pid)].append((p, y))

    # Stratify on each group's majority label
    def group_label(g):
        ys = [y for _, y in by_group[g]]
        return int(sum(ys) * 2 >= len(ys))

    rng = random.Random(seed)
    train, val = [], []
    for label in CLASSES.values():
        keys = sorted(g for g in by_group if group_label(g) == label)
        rng.shuffle(keys)
        n_val = int(len(keys) * val_frac)
        for i, g in enumerate(keys):
            (val if i < n_val else train).extend(by_group[g])
    return train, val


def get_splits(root=DATA_ROOT, val_frac=0.15, seed=42, use_groups=True):
    """Official test set + subject-level train/val split of the official train set.

    If splits/train_groups.json exists (and use_groups=True), near-duplicate scan
    IDs are grouped as one subject before splitting.
    """
    train_all = collect(os.path.join(root, "train"))
    test = collect(os.path.join(root, "test"))
    groups = load_groups() if use_groups else None
    if use_groups and groups is None:
        print("WARNING: no splits/train_groups.json - splitting by scan ID only "
              "(repeat scans of one subject may leak into val).")
    train, val = split_by_patient(train_all, val_frac, seed, groups)
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
