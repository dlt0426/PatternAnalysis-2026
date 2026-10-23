"""Sanity checks: split sizes, class balance and patient-level leakage."""
from collections import Counter
from dataset import get_splits, patient_id

train, val, test = get_splits()

def ids(s):
    return {patient_id(p) for p, _ in s}

for name, s in [("train", train), ("val", val), ("test", test)]:
    n_ad = sum(y for _, y in s)
    per_patient = Counter(patient_id(p) for p, _ in s)
    print(f"{name:5s}: {len(s):6d} slices | {len(ids(s)):4d} patients | "
          f"AD={n_ad} NC={len(s) - n_ad} | slices/patient "
          f"min={min(per_patient.values())} max={max(per_patient.values())}")

print("train∩val :", len(ids(train) & ids(val)))
print("train∩test:", len(ids(train) & ids(test)))
print("val∩test  :", len(ids(val) & ids(test)))
