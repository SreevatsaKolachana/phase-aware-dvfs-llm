#!/bin/bash
#BSUB -J dvfs_bench
#BSUB -n 4
#BSUB -W 180
#BSUB -q gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=32GB]"
#BSUB -o results/logs/job_%J.out
#BSUB -e results/logs/job_%J.err

# ============================================================
# Phase-Aware LLM Inference Benchmark — Hazel HPC Job Script
# ============================================================
# Submit with:  bsub < submit_job.sh
# Monitor with: bjobs
# Check output: cat results/logs/job_<JOBID>.out
# ============================================================

echo "============================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "============================================"

# Load modules
module load cuda/12.1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dvfs

# Verify GPU
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo ""

# Set HuggingFace token (replace with your actual token)
export HF_TOKEN="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"

# Create output directories
mkdir -p results/logs results/figures results/tables data

# Run the benchmark
python run_benchmark.py

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"

