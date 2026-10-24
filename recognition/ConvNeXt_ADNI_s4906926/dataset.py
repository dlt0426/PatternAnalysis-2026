"""
Data loading and subject-level splitting for the ADNI AD vs NC dataset.

File names follow '<scanID>_<sliceIndex>.jpeg' (20 slices per scan).

Split strategy ("official", default)
------------------------------------
The official train/test folders are subject-disjoint, but the official train
folder contains repeat scans of the same subject under different scan IDs
(see audit_split.py). A validation set carved from the train folder therefore
leaks subjects and over-estimates accuracy (val ~0.98 vs test ~0.75).
Hence:
  * train = the whole official train folder
  * val   = half of the official test folder
  * test  = the other half (held out, evaluated once)
The test folder is split by subject group (near-duplicate scan IDs merged,
splits/test_groups.json), stratified by class.

The legacy split ("train": val carved from the train folder) is kept for the
leakage comparison reported in the README.
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
CLASSES = {"NC": 0, "AD": 1}
IMG_SIZE = 224

# Scan-ID -> subject-group mappings produced by `python audit_split.py --make-groups`.
# Scan IDs whose images are near-identical are treated as the same subject.
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
GROUPS_FILE = os.path.join(SPLIT_DIR, "train_groups.json")
TEST_GROUPS_FILE = os.path.join(SPLIT_DIR, "test_groups.json")


def patient_id(path):
    """'.../218391_78.jpeg' -> '218391'"""
    return os.path.basename(path).split("_")[0]


def slice_index(path):
    """'.../218391_78.jpeg' -> 78"""
    return int(os.path.basename(path).split("_")[1].split(".")[0])


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
    """Stratified split of `items` into (rest, val) at the subject level.

    Without `groups`, each scan ID is its own unit. With `groups`, all scan IDs
    mapped to the same group (repeat scans of one subject) move together, so a
    subject can never appear on both sides.
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
    rest, val = [], []
    for label in CLASSES.values():
        keys = sorted(g for g in by_group if group_label(g) == label)
        rng.shuffle(keys)
        n_val = int(len(keys) * val_frac)
        for i, g in enumerate(keys):
            (val if i < n_val else rest).extend(by_group[g])
    return rest, val


def get_splits(root=DATA_ROOT, split="official", val_frac=None, seed=42):
    """Return (train, val, test) lists of (path, label).

    split="official": train = official train; val/test = halves of official test.
    split="train"   : legacy - val = 15% of official train (leaks subjects).
    """
    train_all = collect(os.path.join(root, "train"))
    test_all = collect(os.path.join(root, "test"))

    if split == "official":
        groups = load_groups(TEST_GROUPS_FILE)
        if groups is None:
            print("WARNING: no splits/test_groups.json - splitting the test folder by scan ID only.")
        test, val = split_by_patient(test_all, 0.5 if val_frac is None else val_frac, seed, groups)
        return train_all, val, test

    if split == "train":
        groups = load_groups(GROUPS_FILE)
        if groups is None:
            print("WARNING: no splits/train_groups.json - splitting by scan ID only.")
        train, val = split_by_patient(train_all, 0.15 if val_frac is None else val_frac, seed, groups)
        return train, val, test_all

    raise ValueError(f"Unknown split: {split}")


class ADNIDataset(Dataset):
    """Returns (image, label, scan_id) for each slice.

    context=0: 1-channel image (the slice itself).
    context=1: 3-channel "2.5D" image (slices s-1, s, s+1 of the same scan),
               clamped at the first/last available slice.
    """

    def __init__(self, items, transform=None, context=0):
        if context not in (0, 1):
            raise ValueError("context must be 0 (2D) or 1 (2.5D, 3 slices)")
        self.items = items
        self.transform = transform
        self.context = context
        self.lookup = defaultdict(dict)                   # scan_id -> {slice_index: path}
        for p, _ in items:
            self.lookup[patient_id(p)][slice_index(p)] = p

    def __len__(self):
        return len(self.items)

    def _load(self, path):
        if self.context == 0:
            return Image.open(path).convert("L")
        slices = self.lookup[patient_id(path)]
        lo, hi, s = min(slices), max(slices), slice_index(path)
        chans = [Image.open(slices.get(min(max(s + d, lo), hi), path)).convert("L") for d in (-1, 0, 1)]
        # Merging into one RGB image keeps the random augmentation identical across channels
        return Image.merge("RGB", chans)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = self._load(path)
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


def get_transforms(mean, std, train, channels=1):
    """Augmentation is applied to the training set only; val/test are deterministic."""
    norm = T.Normalize([mean] * channels, [std] * channels)
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),                       # brains are roughly left-right symmetric
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])


def get_loaders(batch_size=64, num_workers=4, root=DATA_ROOT, smoke=False, split="official", context=0):
    """Build train/val/test DataLoaders.

    smoke=True keeps only a small random subset of each split so the whole
    pipeline can be checked in a minute before launching a full run.
    """
    train, val, test = get_splits(root, split)
    if smoke:
        rng = random.Random(0)
        train = rng.sample(train, min(512, len(train)))
        val = rng.sample(val, min(256, len(val)))
        test = rng.sample(test, min(256, len(test)))

    mean, std = compute_mean_std(train, n=200 if smoke else 2000)
    channels = 2 * context + 1

    def make(items, train_mode):
        return DataLoader(
            ADNIDataset(items, get_transforms(mean, std, train_mode, channels), context),
            batch_size=batch_size, shuffle=train_mode,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )

    loaders = {"train": make(train, True), "val": make(val, False), "test": make(test, False)}
    return loaders, (mean, std)
