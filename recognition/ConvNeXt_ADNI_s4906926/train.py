"""
Train, validate and test a model on the ADNI AD vs NC task.

Examples:
    python train.py --model baseline --epochs 20 --out outputs/baseline
    python train.py --model convnext --pretrained --epochs 30 --lr 1e-4 \
        --weight-decay 0.05 --warmup-epochs 3 --label-smoothing 0.1 --out outputs/convnext
    python train.py --model convnext --pretrained --smoke   # quick pipeline check

The checkpoint with the best validation accuracy is saved and evaluated once
on the held-out test set at the end of training.
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")                       # no display on Rangpur compute nodes
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

from dataset import get_loaders, DATA_ROOT
from modules import build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out", default="outputs/baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true", help="ImageNet weights (convnext only)")
    p.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth rate (convnext only)")
    p.add_argument("--warmup-epochs", type=float, default=0, help="linear LR warmup before cosine decay")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 mixed precision on GPU")
    p.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, for debugging")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, amp=False):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.

    Returns mean loss, slice accuracy, and per-slice (prob_AD, label, patient_id)
    so that patient-level metrics can be computed afterwards.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    probs, labels, pids = [], [], []

    with torch.set_grad_enabled(training):
        for x, y, pid in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # bfloat16 autocast: faster on A100 and halves activation memory
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            logits = logits.float()

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()                  # LR is updated every iteration

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            probs.append(torch.softmax(logits, 1)[:, 1].detach().cpu())
            labels.append(y.cpu())
            pids.extend(pid)

    return total_loss / n, correct / n, torch.cat(probs), torch.cat(labels), pids


def make_optimizer(model, lr, weight_decay):
    """AdamW without weight decay on biases, norm weights and Layer Scale (1-D params)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup from ~0 to the base LR, then cosine decay to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def patient_level(probs, labels, pids):
    """Average the AD probability over all slices of each patient."""
    agg = defaultdict(list)
    lab = {}
    for p, y, pid in zip(probs.tolist(), labels.tolist(), pids):
        agg[pid].append(p)
        lab[pid] = y
    keys = sorted(agg)
    p_mean = torch.tensor([sum(agg[k]) / len(agg[k]) for k in keys])
    y = torch.tensor([lab[k] for k in keys])
    return p_mean, y


def summarise(probs, labels):
    """Accuracy, macro-F1 and AUROC from AD probabilities (threshold 0.5)."""
    pred = (probs >= 0.5).long()
    out = {
        "acc": (pred == labels).float().mean().item(),
        "macro_f1": f1_score(labels.tolist(), pred.tolist(), average="macro"),
    }
    out["auroc"] = roc_auc_score(labels.tolist(), probs.tolist()) if labels.unique().numel() > 1 else float("nan")
    return out


def plot_history(history, path):
    """Loss and accuracy curves for train/val."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].set_title("Loss")
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val (slice)")
    ax[1].plot(epochs, history["val_patient_acc"], label="val (patient)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Accuracy")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    loaders, (mean, std) = get_loaders(args.batch_size, args.num_workers, args.data_root, args.smoke)
    print(f"Slices  train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}  "
          f"test={len(loaders['test'].dataset)}  | mean={mean:.4f} std={std:.4f}")

    model = build_model(args.model, pretrained=args.pretrained, drop_path=args.drop_path).to(device)
    print(f"Model: {args.model}  pretrained={args.pretrained}  trainable params: {count_parameters(model):,}")

    amp = device.type == "cuda" and not args.no_amp
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    scheduler = make_scheduler(optimizer, int(args.warmup_epochs * steps_per_epoch),
                               args.epochs * steps_per_epoch)

    history = defaultdict(list)
    best_val, ckpt_path = -1.0, os.path.join(args.out, "best.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, *_ = run_epoch(model, loaders["train"], criterion, device, optimizer, scheduler, amp)
        va_loss, va_acc, va_p, va_y, va_pid = run_epoch(model, loaders["val"], criterion, device, amp=amp)
        va_pat_acc = summarise(*patient_level(va_p, va_y, va_pid))["acc"]

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", va_loss),
                     ("val_acc", va_acc), ("val_patient_acc", va_pat_acc), ("lr", lr_now),
                     ("epoch_time_s", time.time() - t0)]:
            history[k].append(v)
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} patient-acc {va_pat_acc:.4f} | "
              f"lr {lr_now:.2e} | {time.time() - t0:.0f}s")

        # Model selection uses the validation set only (never the test set).
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "mean": mean, "std": std, "epoch": epoch, "args": vars(args)}, ckpt_path)

        plot_history(history, os.path.join(args.out, "curves.png"))

    # ---- Final, one-off evaluation on the held-out test set ----
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, _, te_p, te_y, te_pid = run_epoch(model, loaders["test"], criterion, device, amp=amp)
    results = {
        "model": args.model,
        "best_epoch": ckpt["epoch"],
        "params": count_parameters(model),
        "test_slice": summarise(te_p, te_y),
        "test_patient": summarise(*patient_level(te_p, te_y, te_pid)),
    }
    if device.type == "cuda":
        results["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 1024 ** 2

    print("\n=== Test results (best val checkpoint, epoch %d) ===" % ckpt["epoch"])
    for level in ("test_slice", "test_patient"):
        r = results[level]
        print(f"{level:13s} acc {r['acc']:.4f}  macro-F1 {r['macro_f1']:.4f}  AUROC {r['auroc']:.4f}")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"results": results, "history": history, "args": vars(args)}, f, indent=2)
    print(f"Saved checkpoint, curves.png and results.json to {args.out}/")


if __name__ == "__main__":
    main()
