# Feasibility Review — ConvNeXt for Alzheimer's Disease Classification (ADNI)

**Student:** s4906926 · **Domain:** Classification · **Difficulty:** Hard · **Dataset:** ADNI AD vs NC (`/home/groups/comp3710/ADNI`)

## 1. User Need, Scope & Acceptance Criteria

**User:** a radiology screening team that wants an assistant to pre-sort T1-weighted brain MRIs into Alzheimer's Disease (AD) vs Cognitively Normal (NC), auto-reporting confident cases and **routing uncertain cases to a specialist**. **Prototype scope:** 2D axial slices → patient-level decision by averaging slice probabilities. It is a triage aid, not a diagnostic device.

**Acceptance criteria** (all measured on the held-out official test set, 450 patients, never used for model selection):

1. **Accuracy:** patient-level test accuracy ≥ 0.80, with macro-F1 and AUROC reported, and better than the baseline CNN on the same split.
2. **Calibration (dilemma):** Expected Calibration Error (ECE) ≤ 0.10, shown with a reliability diagram for both ConvNeXt and the baseline.
3. **Reject option (dilemma):** with a confidence threshold chosen on the *validation* set that defers ≤ 20% of patients to a specialist, accuracy on the auto-handled patients ≥ 0.90.
4. **Leakage-free evaluation:** zero patient-ID overlap between train / val / test, verified by an automated check.
5. **Resource constraint:** peak training VRAM ≤ 16 GB (batch 64, 224×224) and inference ≤ 1 s per patient (20 slices) on one Rangpur GPU; parameter count reported.

## 2. Model Choice & Course Concepts

- **Data:** official `train/` split into train/val **by patient** (stratified, 15% val, seed 42); official `test/` held out. Grayscale 224×224, normalised with training-set mean/std; flip + small affine augmentation on training only.
- **Model:** **ConvNeXt** (Liu et al., CVPR 2022) — a CNN "modernised" with Transformer design choices: patchify stem, 7×7 depthwise convolutions, inverted bottleneck, LayerNorm and GELU. It builds directly on course concepts (convolution, residual connections, normalisation) while matching ViT-level accuracy with CNN inductive biases, which suit a ~900-patient dataset.
- **Baseline:** 4-block plain CNN (Conv-BN-ReLU-MaxPool, 0.39 M parameters), same splits and training loop.
- **Course concepts:** CNNs/ResNets, regularisation (weight decay, dropout, augmentation), AdamW with cosine LR schedule, train/val/test methodology, evaluation beyond accuracy.

## 3. Preliminary Feasibility Evidence

**Data audit** (`check_split.py`):

| Split | Slices | Patients | AD / NC slices | Slices per patient |
|---|---|---|---|---|
| Train | 18,300 | 915 | 8,840 / 9,460 | 20 |
| Val | 3,220 | 161 | 1,560 / 1,660 | 20 |
| Test | 9,000 | 450 | 4,460 / 4,540 | 20 |

Patient overlap train∩val = train∩test = val∩test = **0**. Classes are balanced (≈48% AD), so accuracy is a meaningful headline metric.

**Baseline smoke test** (BaselineCNN, [N] epochs, [GPU type]):

| Level | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|
| Slice (test) | [ ] | [ ] | [ ] |
| Patient (test) | [ ] | [ ] | [ ] |

Best val accuracy [ ] at epoch [ ]; peak VRAM [ ] MB; ~[ ] s/epoch. **Observation:** [e.g. train acc reaches X while val plateaus at Y → overfitting; patient averaging adds +Z].

## 4. Risks, Budget & Fallback

| Risk | Mitigation |
|---|---|
| 0.80 is hard for single 2D slices (subtle atrophy) | Patient-level averaging; stronger augmentation; 2.5D input (3 adjacent slices as channels) |
| Overfitting with ~900 training patients | Small variant (ConvNeXt-Tiny or smaller), stochastic depth, early stopping on val |
| Overconfident wrong predictions | Temperature scaling fitted on val; reject-option threshold from val |
| Residual leakage: file ID may be a *scan* ID, not a subject ID | Cannot be verified from filenames; stated as a limitation of the reported test accuracy |

**Compute budget:** ≈ [ ] GPU-hours (baseline ≈ [ ] min; ConvNeXt-Tiny estimated ≈ 1–2 h per 50-epoch run; ~10 runs for tuning).
**Next experiment:** implement ConvNeXt-Tiny in `modules.py`, train from scratch with identical settings, compare with the baseline.
**Fallback:** if from-scratch ConvNeXt stays below 0.80, (a) initialise from ImageNet-pretrained weights *if permitted by teaching staff*, or (b) switch to 2.5D input with a smaller ConvNeXt; the baseline + calibration/reject-option analysis remains a complete deliverable either way.
