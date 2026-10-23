"""
Nearest-neighbour audit of the patient-level splits.

Hypothesis: ADNI is longitudinal, so one subject can appear under several
scan IDs. If two scans of the same subject land in our train and val splits,
validation accuracy is inflated, while the official test split (separated by
subject) is not affected.

Method (model-independent, raw pixels only):
  * Each scan ID -> fingerprint = all 20 slices, resized to 32x32, flattened,
    zero-mean / unit-norm.
  * For every val and test scan, find its most similar TRAIN scan (cosine
    similarity) and record that similarity and whether the labels agree.
  * If val scans have far closer train neighbours than test scans, the val
    split contains repeat scans of training subjects.

Usage:
    python audit_split.py                    # writes outputs/audit/
"""
import argparse
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_splits, patient_id, DATA_ROOT

SIZE = 32


def fingerprints(items):
    """Return (ids, labels, (N, D) unit-norm fingerprint matrix) for a split."""
    by_pid = defaultdict(list)
    label = {}
    for p, y in items:
        by_pid[patient_id(p)].append(p)
        label[patient_id(p)] = y
    ids = sorted(by_pid)
    feats = []
    for pid in ids:
        # sort slices by slice index so fingerprints are aligned across scans
        paths = sorted(by_pid[pid], key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        slices = []
        for p in paths:
            img = Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
            slices.append(torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float())
        v = torch.cat(slices)
        v = v - v.mean()
        feats.append(v / (v.norm() + 1e-8))
    return ids, torch.tensor([label[i] for i in ids]), torch.stack(feats)


def nearest(query, ref, exclude_self=False):
    """Max cosine similarity and index of the nearest ref row for each query row."""
    sim = query @ ref.T
    if exclude_self:
        sim.fill_diagonal_(-1.0)
    return sim.max(dim=1)


def describe(name, s, agree):
    q = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9]))
    return {"split": name, "n": len(s), "p10": q[0].item(), "median": q[1].item(), "p90": q[2].item(),
            "nn_label_agreement": agree.float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default="outputs/audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train, val, test = get_splits(args.data_root)
    print("Building fingerprints (CPU, a few minutes)...")
    tr_ids, tr_y, tr_f = fingerprints(train)
    va_ids, va_y, va_f = fingerprints(val)
    te_ids, te_y, te_f = fingerprints(test)

    s_tt, i_tt = nearest(tr_f, tr_f, exclude_self=True)   # train vs other train (reference)
    s_va, i_va = nearest(va_f, tr_f)
    s_te, i_te = nearest(te_f, tr_f)

    rows = [describe("train->train", s_tt, tr_y[i_tt] == tr_y),
            describe("val->train", s_va, tr_y[i_va] == va_y),
            describe("test->train", s_te, tr_y[i_te] == te_y)]

    # "Suspicious" = closer to a train scan than 99% of test scans are.
    thr = torch.quantile(s_te, 0.99).item()
    for r, s in zip(rows, (s_tt, s_va, s_te)):
        r["frac_above_test_p99"] = (s > thr).float().mean().item()

    print(f"\n{'split':13s} {'n':>5s} {'p10':>7s} {'median':>7s} {'p90':>7s} "
          f"{'>test p99':>10s} {'1-NN label agree':>17s}")
    for r in rows:
        print(f"{r['split']:13s} {r['n']:5d} {r['p10']:7.3f} {r['median']:7.3f} {r['p90']:7.3f} "
              f"{r['frac_above_test_p99']:10.1%} {r['nn_label_agreement']:17.1%}")
    print(f"(test p99 similarity threshold = {thr:.3f})")

    # Most similar val/train pairs for visual inspection
    top = torch.argsort(s_va, descending=True)[:10]
    pairs = [{"val_id": va_ids[k], "train_id": tr_ids[i_va[k]], "similarity": s_va[k].item()} for k in top]
    print("\nMost similar val/train pairs:")
    for p in pairs:
        print(f"  val {p['val_id']}  <->  train {p['train_id']}   sim={p['similarity']:.4f}")

    # Histogram
    plt.figure(figsize=(7, 4))
    bins = torch.linspace(min(s.min().item() for s in (s_tt, s_va, s_te)), 1.0, 60).tolist()
    for s, lab in [(s_tt, "train → nearest other train"), (s_va, "val → nearest train"),
                   (s_te, "test → nearest train")]:
        plt.hist(s.tolist(), bins=bins, alpha=0.5, density=True, label=lab)
    plt.axvline(thr, color="k", ls="--", lw=1, label="test 99th percentile")
    plt.xlabel("cosine similarity to nearest training scan (raw pixels)")
    plt.ylabel("density"); plt.legend(); plt.title("Nearest-neighbour audit of splits")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "nn_audit.png"), dpi=150)

    with open(os.path.join(args.out, "nn_audit.json"), "w") as f:
        json.dump({"summary": rows, "threshold_test_p99": thr, "top_val_pairs": pairs}, f, indent=2)
    print(f"\nSaved nn_audit.png and nn_audit.json to {args.out}/")


if __name__ == "__main__":
    main()
