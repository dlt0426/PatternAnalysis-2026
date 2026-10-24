"""Sanity checks: split sizes, class balance and subject-level leakage.

Usage: python check_split.py [official|train]
"""
import sys
from collections import Counter
from dataset import get_splits, patient_id, load_groups, GROUPS_FILE, TEST_GROUPS_FILE

split = sys.argv[1] if len(sys.argv) > 1 else "official"
train, val, test = get_splits(split=split)
groups = {**(load_groups(GROUPS_FILE) or {}), **(load_groups(TEST_GROUPS_FILE) or {})}

def ids(s):
    return {patient_id(p) for p, _ in s}

def subjects(s):
    return {groups.get(pid, pid) for pid in ids(s)}

print(f"split = {split}")
for name, s in [("train", train), ("val", val), ("test", test)]:
    n_ad = sum(y for _, y in s)
    per_scan = Counter(patient_id(p) for p, _ in s)
    print(f"{name:5s}: {len(s):6d} slices | {len(ids(s)):4d} scan IDs | {len(subjects(s)):4d} subject groups | "
          f"AD={n_ad} NC={len(s) - n_ad} | slices/scan min={min(per_scan.values())} max={max(per_scan.values())}")

for a, b, sa, sb in [("train", "val", train, val), ("train", "test", train, test), ("val", "test", val, test)]:
    print(f"overlap {a}∩{b}:  scan IDs {len(ids(sa) & ids(sb))}  subject groups {len(subjects(sa) & subjects(sb))}")
