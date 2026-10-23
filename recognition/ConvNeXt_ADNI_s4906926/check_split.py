"""Sanity checks: split sizes, class balance and subject-level leakage."""
from collections import Counter
from dataset import get_splits, patient_id, load_groups

train, val, test = get_splits()
groups = load_groups() or {}

def ids(s):
    return {patient_id(p) for p, _ in s}

def subjects(s):
    return {groups.get(pid, pid) for pid in ids(s)}

for name, s in [("train", train), ("val", val), ("test", test)]:
    n_ad = sum(y for _, y in s)
    per_patient = Counter(patient_id(p) for p, _ in s)
    print(f"{name:5s}: {len(s):6d} slices | {len(ids(s)):4d} scan IDs | {len(subjects(s)):4d} subject groups | "
          f"AD={n_ad} NC={len(s) - n_ad} | slices/scan min={min(per_patient.values())} max={max(per_patient.values())}")

print("scan-ID overlap   train∩val:", len(ids(train) & ids(val)), " train∩test:", len(ids(train) & ids(test)),
      " val∩test:", len(ids(val) & ids(test)))
print("subject overlap   train∩val:", len(subjects(train) & subjects(val)))
