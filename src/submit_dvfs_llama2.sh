#!/bin/bash
#BSUB -J dvfs_llama2
#BSUB -n 4
#BSUB -W 240
#BSUB -q gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o results_llama2/logs/dvfs_llama2_%J.out
#BSUB -e results_llama2/logs/dvfs_llama2_%J.err

echo "============================================"
echo "DVFS Llama-2-7B Job started: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "============================================"

module load cuda/12.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dvfs

# Create output dirs
mkdir -p results_llama2/logs results_llama2/figures results_llama2/tables

# Verify GPU
nvidia-smi

# Verify model exists
if [ ! -d "models/Llama-2-7b-chat-hf" ]; then
    echo "ERROR: models/Llama-2-7b-chat-hf not found!"
    exit 1
fi

echo "Starting Llama-2-7B DVFS benchmark..."
python run_dvfs_llama2.py

echo "============================================"
echo "Job completed: $(date)"
echo "============================================"
