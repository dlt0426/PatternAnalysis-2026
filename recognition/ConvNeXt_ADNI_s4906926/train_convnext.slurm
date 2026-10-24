#!/bin/bash
#SBATCH --job-name=adni_convnext
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out   # stdout/stderr -> logs/adni_convnext_<jobid>.out

# Usage: sbatch train_convnext.slurm [extra train.py args], e.g.
#   sbatch train_convnext.slurm --out outputs/convnext_2d
#   sbatch train_convnext.slurm --context-slices 1 --out outputs/convnext_25d
# Later arguments override the defaults below.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate comp3710

cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_s4906926
nvidia-smi
python train.py --model convnext --pretrained \
    --epochs 30 --batch-size 64 --lr 1e-4 --weight-decay 0.05 \
    --warmup-epochs 3 --label-smoothing 0.1 --drop-path 0.1 \
    --num-workers 8 "$@"
