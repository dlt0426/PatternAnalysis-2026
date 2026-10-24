"""
Evaluate trained checkpoints and produce the analysis used in the README.

For each checkpoint this script:
  1. Runs inference on the validation and held-out test sets.
  2. Calibrates probabilities with temperature scaling fitted on VALIDATION.
  3. Chooses the patient-level decision threshold on VALIDATION
     (max balanced accuracy) and applies it unchanged to TEST.
  4. Reports slice- and patient-level accuracy, sensitivity, specificity,
     macro-F1 and AUROC with 95% bootstrap confidence intervals.
  5. Measures calibration (ECE, reliability diagram).
  6. Selective prediction: patients whose probability is close to the
     threshold are deferred to a clinician. The deferral cut-off is chosen on
     validation for a target accuracy and then applied to test.
  7. Measures resources (parameters, inference latency, peak GPU memory).
  8. Saves Grad-CAM overlays of the most confident errors for failure analysis.

Usage:
    python predict.py outputs/convnext_2d/best.pt
    python predict.py outputs/baseline_2d/best.pt outputs/convnext_2d/best.pt   # comparison table

Outputs go to <checkpoint dir>/eval/.
NumPy is allowed in this script (visualisation), but is not needed.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

from dataset import ADNIDataset, get_splits, get_transforms, patient_id, slice_index, DATA_ROOT
from modules import build_model, count_parameters, ConvNeXt, BaselineCNN


# ---------------------------------------------------------------------------
# Loading and inference
# ---------------------------------------------------------------------------
def load_checkpoint(path, device):
    """Rebuild the model exactly as it was trained (architecture, input channels)."""
    ckpt = torch.load(path, map_location=device)
    a = ckpt.get("args", {})
    model = build_model(ckpt["model"], pretrained=False, drop_path=a.get("drop_path", 0.1),
                        in_chans=2 * a.get("context_slices", 0) + 1)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt, a


@torch.no_grad()
def infer(model, loader, device):
    """Slice-level logits, labels, scan IDs and file paths for one split."""
    logits, labels, pids, paths = [], [], [], []
    for x, y, pid in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
        pids.extend(pid)
    paths = [p for p, _ in loader.dataset.items]
    return {"logits": torch.cat(logits), "labels": torch.cat(labels), "pids": pids, "paths": paths}


def patient_probs(prob_ad, labels, pids):
    """Average slice-level AD probability per scan -> (ids, prob, label)."""
    agg, lab = defaultdict(list), {}
    for p, y, pid in zip(prob_ad.tolist(), labels.tolist(), pids):
        agg[pid].append(p)
        lab[pid] = y
    ids = sorted(agg)
    return ids, torch.tensor([sum(agg[k]) / len(agg[k]) for k in ids]), torch.tensor([lab[k] for k in ids])


# ---------------------------------------------------------------------------
# Calibration and thresholds (fitted on validation only)
# ---------------------------------------------------------------------------
def fit_temperature(logits, labels):
    """Temperature scaling: one scalar T > 0 minimising validation NLL of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def best_threshold(prob, labels):
    """Threshold maximising balanced accuracy (ties -> closest to 0.5)."""
    best = (-1.0, 0.0, 0.5)
    for t in sorted(set(prob.tolist()) | {0.5}):
        pred = (prob >= t).long()
        sens = ((pred == 1) & (labels == 1)).sum().item() / max(1, (labels == 1).sum().item())
        spec = ((pred == 0) & (labels == 0)).sum().item() / max(1, (labels == 0).sum().item())
        cand = ((sens + spec) / 2, -abs(t - 0.5), t)
        best = max(best, cand)
    return best[2]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(prob, labels, t=0.5):
    pred = (prob >= t).long()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    return {
        "acc": (tp + tn) / len(labels),
        "sensitivity_AD": tp / max(1, tp + fn),
        "specificity_NC": tn / max(1, tn + fp),
        "precision_AD": tp / max(1, tp + fp),
        "macro_f1": f1_score(labels.tolist(), pred.tolist(), average="macro"),
        "auroc": roc_auc_score(labels.tolist(), prob.tolist()),
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
    }


def bootstrap_ci(prob, labels, t, n=1000, seed=0):
    """95% percentile bootstrap CI for patient-level accuracy and AUROC."""
    g = torch.Generator().manual_seed(seed)
    accs, aucs = [], []
    for _ in range(n):
        idx = torch.randint(len(labels), (len(labels),), generator=g)
        p, y = prob[idx], labels[idx]
        if y.unique().numel() < 2:
            continue
        accs.append(((p >= t).long() == y).float().mean().item())
        aucs.append(roc_auc_score(y.tolist(), p.tolist()))

    def ci(v):
        v = torch.tensor(v)
        return [torch.quantile(v, 0.025).item(), torch.quantile(v, 0.975).item()]
    return {"acc_95ci": ci(accs), "auroc_95ci": ci(aucs)}


def ece(prob, labels, t=0.5, n_bins=10):
    """Expected calibration error of the predicted class' confidence."""
    pred = (prob >= t).long()
    conf = torch.where(pred == 1, prob, 1 - prob)
    correct = (pred == labels).float()
    bins = torch.linspace(0.5, 1.0, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = (correct[m].mean() - conf[m].mean()).abs().item()
        total += m.float().mean().item() * gap
        rows.append((conf[m].mean().item(), correct[m].mean().item(), m.sum().item()))
    return total, rows


def confidence(prob, t):
    """Normalised distance from the decision threshold, in [0, 1] on both sides.

    With t != 0.5 the raw |p - t| is asymmetric (an NC prediction can be
    further from t than an AD prediction), so each side is rescaled.
    """
    return torch.where(prob >= t, (prob - t) / max(1e-8, 1 - t), (t - prob) / max(1e-8, t))


def selective(prob, labels, t):
    """Coverage/accuracy when deferring the patients closest to the threshold."""
    margin = confidence(prob, t)
    order = torch.argsort(margin, descending=True)              # most confident first
    correct = ((prob >= t).long() == labels).float()[order]
    n = len(labels)
    cov = torch.arange(1, n + 1).float() / n
    acc = torch.cumsum(correct, 0) / torch.arange(1, n + 1)
    return margin[order], cov, acc


def deferral_cutoff(prob, labels, t, target):
    """Smallest confidence such that val accuracy on retained patients >= target (max coverage)."""
    margins, cov, acc = selective(prob, labels, t)
    ok = (acc >= target).nonzero()
    if len(ok) == 0:
        return None
    k = ok.max().item()
    return margins[k].item()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@torch.no_grad()
def latency(model, in_chans, device, batch, reps=50):
    x = torch.randn(batch, in_chans, 224, 224, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000                 # ms per batch


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def gradcam(model, x, target_class):
    """Grad-CAM heatmaps (B, 224, 224) in [0, 1] for `target_class` per sample."""
    layer = model.stages[-1] if isinstance(model, ConvNeXt) else model.features[-1]
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    model.zero_grad()
    with torch.enable_grad():
        out = model(x.detach().clone().requires_grad_(True))
        out.gather(1, target_class.view(-1, 1)).sum().backward()
    h1.remove(); h2.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)              # channel importance
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cam.detach()


def failure_figure(model, dataset, cases, device, path):
    """One row per patient: 5 slices across the scan with Grad-CAM for the predicted class."""
    if not cases:
        print(f"  (no cases for {os.path.basename(path)})")
        return
    by_pid = defaultdict(list)
    for i, (p, _) in enumerate(dataset.items):
        by_pid[patient_id(p)].append(i)
    fig, axes = plt.subplots(len(cases), 5, figsize=(12, 2.6 * len(cases)), squeeze=False)
    for r, c in enumerate(cases):
        idx = sorted(by_pid[c["id"]], key=lambda i: slice_index(dataset.items[i][0]))
        pick = [idx[int(j * (len(idx) - 1) / 4)] for j in range(5)]
        x = torch.stack([dataset[i][0] for i in pick]).to(device)
        cam = gradcam(model, x, torch.full((5,), c["pred"], device=device)).cpu()
        mid = x.shape[1] // 2                                        # centre slice for 2.5D input
        for k in range(5):
            ax = axes[r, k]
            ax.imshow(x[k, mid].cpu(), cmap="gray")
            ax.imshow(cam[k], cmap="jet", alpha=0.35)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"slice {slice_index(dataset.items[pick[k]][0])}", fontsize=8)
        names = ["NC", "AD"]
        axes[r, 0].set_ylabel(f"{c['id']}\ntrue {names[c['label']]} / pred {names[c['pred']]}\n"
                              f"P(AD)={c['prob']:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(ckpt_path, args, device):
    model, ckpt, a = load_checkpoint(ckpt_path, device)
    name = os.path.basename(os.path.dirname(ckpt_path))
    out = os.path.join(os.path.dirname(ckpt_path), "eval")
    os.makedirs(out, exist_ok=True)
    context = a.get("context_slices", 0)
    split = a.get("split", "train")                                  # checkpoints before the split change
    if split != "official":
        print(f"NOTE: {name} was trained with split='{split}'; evaluating on the official val/test halves.")

    _, val_items, test_items = get_splits(args.data_root, "official")
    tf = get_transforms(ckpt["mean"], ckpt["std"], train=False, channels=2 * context + 1)
    ds = {k: ADNIDataset(v, tf, context) for k, v in [("val", val_items), ("test", test_items)]}
    res = {k: infer(model, DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers), device)
           for k, d in ds.items()}

    # 1) temperature + threshold on validation
    T = fit_temperature(res["val"]["logits"], res["val"]["labels"])
    for r in res.values():
        r["prob_raw"] = torch.softmax(r["logits"], 1)[:, 1]
        r["prob_cal"] = torch.softmax(r["logits"] / T, 1)[:, 1]
        r["pat"] = {k: patient_probs(r[k], r["labels"], r["pids"]) for k in ("prob_raw", "prob_cal")}
    _, vp, vy = res["val"]["pat"]["prob_cal"]
    t_star = best_threshold(vp, vy)

    # 2) test metrics
    ids, tp, ty = res["test"]["pat"]["prob_cal"]
    _, tp_raw, _ = res["test"]["pat"]["prob_raw"]
    report = {
        "checkpoint": ckpt_path, "model": ckpt["model"], "trained_split": split, "context_slices": context,
        "temperature": T, "threshold_from_val": t_star,
        "val_patient@0.5": metrics(vp, vy, 0.5),
        "test_slice@0.5": metrics(res["test"]["prob_raw"], res["test"]["labels"], 0.5),
        "test_patient@0.5": {**metrics(tp_raw, ty, 0.5), **bootstrap_ci(tp_raw, ty, 0.5)},
        "test_patient@val_threshold": {**metrics(tp, ty, t_star), **bootstrap_ci(tp, ty, t_star)},
    }

    # 3) calibration
    ece_raw, rows_raw = ece(tp_raw, ty, 0.5)
    ece_cal, rows_cal = ece(tp, ty, t_star)
    report["test_patient_ECE"] = {"before_temperature": ece_raw, "after_temperature": ece_cal}
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    for rows, lab in [(rows_raw, f"raw (ECE {ece_raw:.3f})"), (rows_cal, f"temp-scaled (ECE {ece_cal:.3f})")]:
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label=lab)
    ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy"); ax.set_title(f"{name}: test reliability (patient)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "reliability.png"), dpi=150); plt.close(fig)

    # 4) selective prediction / deferral to clinician
    sel = {}
    for target in args.targets:
        cut = deferral_cutoff(vp, vy, t_star, target)
        if cut is None:
            sel[str(target)] = "unreachable on validation"
            continue
        keep = confidence(tp, t_star) >= cut
        acc_keep = (((tp[keep] >= t_star).long() == ty[keep]).float().mean().item()) if keep.any() else float("nan")
        sel[str(target)] = {"margin_cutoff": cut, "test_coverage": keep.float().mean().item(),
                            "test_deferred": 1 - keep.float().mean().item(), "test_acc_automated": acc_keep}
    report["selective_prediction"] = sel
    _, cov_t, acc_t = selective(tp, ty, t_star)
    _, cov_v, acc_v = selective(vp, vy, t_star)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(cov_v, acc_v, label="validation"); ax.plot(cov_t, acc_t, label="test")
    ax.set_xlabel("coverage (fraction decided automatically)"); ax.set_ylabel("accuracy on automated cases")
    ax.set_title(f"{name}: defer low-confidence patients"); ax.grid(alpha=0.3); ax.legend()
    ax.set_ylim(0.6, 1.01); fig.tight_layout()
    fig.savefig(os.path.join(out, "selective.png"), dpi=150); plt.close(fig)

    # 5) resources
    in_chans = 2 * context + 1
    report["resources"] = {"params": count_parameters(model),
                           "latency_ms_1_slice": latency(model, in_chans, device, 1),
                           "latency_ms_1_patient_20_slices": latency(model, in_chans, device, 20),
                           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        latency(model, in_chans, device, 64, reps=5)
        report["resources"]["peak_inference_mem_MB_batch64"] = torch.cuda.max_memory_allocated() / 1024 ** 2

    # 6) failure cases: most confident wrong patients (+ one confident correct per class)
    pred = (tp >= t_star).long()
    margin = confidence(tp, t_star)
    wrong = [i for i in torch.argsort(margin, descending=True).tolist() if pred[i] != ty[i]][:args.n_failures]
    right = []
    for cls in (1, 0):
        cand = [i for i in torch.argsort(margin, descending=True).tolist() if pred[i] == ty[i] == cls]
        right += cand[:1]
    case = lambda i: {"id": ids[i], "label": ty[i].item(), "pred": pred[i].item(), "prob": tp[i].item()}
    report["failure_cases"] = [case(i) for i in wrong]
    failure_figure(model, ds["test"], [case(i) for i in wrong], device, os.path.join(out, "failures.png"))
    failure_figure(model, ds["test"], [case(i) for i in right], device, os.path.join(out, "correct_examples.png"))

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(name, report)
    print(f"Saved metrics.json, reliability.png, selective.png, failures.png, correct_examples.png to {out}/\n")
    return name, report


def print_report(name, r):
    def row(label, m):
        ci = f"  acc 95% CI [{m['acc_95ci'][0]:.3f}, {m['acc_95ci'][1]:.3f}]" if "acc_95ci" in m else ""
        print(f"  {label:28s} acc {m['acc']:.4f}  sens {m['sensitivity_AD']:.3f}  spec {m['specificity_NC']:.3f}  "
              f"F1 {m['macro_f1']:.3f}  AUROC {m['auroc']:.4f}{ci}")
    print(f"\n===== {name}  (T={r['temperature']:.2f}, threshold from val={r['threshold_from_val']:.3f}) =====")
    row("val patient @0.5", r["val_patient@0.5"])
    row("test slice @0.5", r["test_slice@0.5"])
    row("test patient @0.5", r["test_patient@0.5"])
    row("test patient @val threshold", r["test_patient@val_threshold"])
    print(f"  confusion (test patient @val threshold): {r['test_patient@val_threshold']['confusion']}")
    e = r["test_patient_ECE"]
    print(f"  ECE before/after temperature scaling: {e['before_temperature']:.3f} / {e['after_temperature']:.3f}")
    for target, s in r["selective_prediction"].items():
        if isinstance(s, str):
            print(f"  defer for target acc {target}: {s}")
        else:
            print(f"  defer for target acc {target}: defer {s['test_deferred']:.1%} of test patients -> "
                  f"automated acc {s['test_acc_automated']:.3f}")
    res = r["resources"]
    print(f"  params {res['params']:,} | latency {res['latency_ms_1_slice']:.1f} ms/slice, "
          f"{res['latency_ms_1_patient_20_slices']:.1f} ms/patient ({res['device']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.85, 0.9])
    ap.add_argument("--n-failures", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [evaluate(c, args, device) for c in args.checkpoints]
    if len(results) > 1:
        print("\n===== Comparison (test, patient level, threshold chosen on val) =====")
        print(f"{'model':22s} {'acc':>7s} {'95% CI':>16s} {'AUROC':>7s} {'sens':>6s} {'spec':>6s} "
              f"{'ECE':>6s} {'params':>11s} {'ms/patient':>11s}")
        for name, r in results:
            m = r["test_patient@val_threshold"]
            print(f"{name:22s} {m['acc']:7.4f} [{m['acc_95ci'][0]:.3f},{m['acc_95ci'][1]:.3f}] {m['auroc']:7.4f} "
                  f"{m['sensitivity_AD']:6.3f} {m['specificity_NC']:6.3f} "
                  f"{r['test_patient_ECE']['after_temperature']:6.3f} {r['resources']['params']:11,d} "
                  f"{r['resources']['latency_ms_1_patient_20_slices']:11.1f}")


if __name__ == "__main__":
    main()
