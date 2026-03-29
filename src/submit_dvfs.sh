#!/bin/bash
#BSUB -J dvfs_sim
#BSUB -n 4
#BSUB -W 240
#BSUB -q gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o results/logs/dvfs_job_%J.out
#BSUB -e results/logs/dvfs_job_%J.err

echo "============================================"
echo "DVFS Simulation Job started: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "============================================"

module load cuda/12.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dvfs

nvidia-smi
echo ""

export HF_TOKEN="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"
mkdir -p results/logs results/figures results/tables data

python run_dvfs.py

echo ""
echo "============================================"
echo "DVFS Simulation Job finished: $(date)"
echo "============================================"

