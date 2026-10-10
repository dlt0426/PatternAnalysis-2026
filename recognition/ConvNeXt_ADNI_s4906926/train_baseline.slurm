#!/bin/bash
#SBATCH --job-name=adni_baseline
#SBATCH --partition=a100          # check available GPU partitions with `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out   # stdout/stderr -> logs/adni_baseline_<jobid>.out

# Activate the conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate comp3710

cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_s4906926
nvidia-smi
python train.py --model baseline --epochs 20 --batch-size 64 --lr 1e-3 --out outputs/baseline
