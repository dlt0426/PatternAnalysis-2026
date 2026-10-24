#!/bin/bash
#SBATCH --job-name=adni_baseline
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out   # stdout/stderr -> logs/adni_baseline_<jobid>.out

# Usage: sbatch train_baseline.slurm [extra train.py args], e.g.
#   sbatch train_baseline.slurm --out outputs/baseline_2d
#   sbatch train_baseline.slurm --context-slices 1 --out outputs/baseline_25d
# Later arguments override the defaults below.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate comp3710

cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_s4906926
nvidia-smi
python train.py --model baseline --epochs 20 --batch-size 64 --lr 1e-3 \
    --num-workers 8 "$@"
