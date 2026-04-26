#!/bin/bash
#BSUB -J dvfs_adapt_v2
#BSUB -n 4
#BSUB -W 420
#BSUB -q gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o results_adaptive_v2/logs/adaptive_v2_%J.out
#BSUB -e results_adaptive_v2/logs/adaptive_v2_%J.err

echo "============================================"
echo "Adaptive DVFS v2 started: $(date)"
echo "Host: $(hostname)"
echo "============================================"

module load cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dvfs

mkdir -p results_adaptive_v2/logs results_adaptive_v2/figures results_adaptive_v2/tables

nvidia-smi

python run_adaptive_v2.py

echo "============================================"
echo "Job completed: $(date)"
echo "============================================"
