# Phase-Aware DVFS Scheduling for Energy-Efficient LLM Inference

**Author:** Sreevatsa Kolachana (svkolach@ncsu.edu)  
**Course:** ECE 592 — Generative AI, NC State University  
**Advisor:** Prof. Kaixiong Zhou  
**Date:** April 2026

## Overview

This project implements phase-aware Dynamic Voltage and Frequency Scaling (DVFS) strategies for reducing GPU energy consumption during LLM inference. By decomposing `model.generate()` into explicit prefill and decode phases, we insert calibrated sleep delays during the memory-bound decode phase to trigger GPU idle power states — achieving **5.7–8.7% energy savings** on Qwen2.5-7B without exceeding latency SLOs.

We implement and compare three scheduling policies:
- **Default:** Full GPU speed, no delays (baseline)
- **Static Phase-Aware:** Fixed 5ms delay after every decode token
- **Adaptive Phase-Aware:** Dynamic per-token delay based on SLO slack: `d_i = clamp(α × (T_budget − t_i), 0, d_max)`

All experiments run on NVIDIA A100 GPUs on NCSU's Hazel HPC cluster **without root access**, using software-only throttling via `time.sleep()`.

## Key Results

| Policy | Short Prompts | Medium Prompts | Long Prompts |
|--------|:---:|:---:|:---:|
| Static (5ms) | +5.7% saving | +6.7% saving | +7.5% saving |
| Adaptive (α=0.12) | +6.2% saving | +8.7% saving | +6.1% saving |

- Average GPU power drops ~19% (114W → 93W) under throttling
- All TBT P95 values remain under 75ms (SLO = 100ms)
- Cross-architecture analysis on 3 models shows effectiveness correlates with baseline GPU power

## Repository Structure

```
phase-aware-dvfs-llm/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── run_benchmark.py               # Baseline phase-separated benchmark
│   ├── run_dvfs.py                    # Static DVFS experiments (Default/Static/Phase-Aware)
│   ├── run_adaptive_v2.py             # Adaptive DVFS experiments (calibrated α sweep)
│   ├── run_dvfs_llama2.py             # Llama-2-7B DVFS benchmark
│   ├── submit_job.sh                  # LSF job submission for baseline
│   ├── submit_dvfs.sh                 # LSF job submission for DVFS
│   ├── submit_adaptive_v2.sh          # LSF job submission for adaptive
│   └── submit_dvfs_llama2.sh          # LSF job submission for Llama-2
│
├── analysis/                          # Cross-architecture analysis scripts
│   ├── compute_arithmetic_intensity.py # Computes M/C ratio for each model
│   ├── generate_gain_3model.py        # Gain vs architecture plots (3 models)
│   ├── generate_gain_vs_mc.py         # Gain vs M/C ratio analysis
│   ├── plot_correlations.py           # Energy-time, power-throughput plots
│   └── model_architectures.json       # Architecture params + experimental data
│
├── data/                              # Dataset files
│   ├── sharegpt_bucketed_n40.json     # 40 prompts/bucket for baseline
│   └── sharegpt_bucketed_n20_dvfs.json # 20 prompts/bucket for DVFS
│
├── results/                           # Experimental results
│   ├── figures/                       # All generated plots
│   │   ├── dvfs_comparison_Qwen2.5-7B.png
│   │   ├── dvfs_sweep_Qwen2.5-7B.png
│   │   ├── gain_3model_main.png
│   │   ├── energy_vs_time.png
│   │   ├── delay_vs_compute.png
│   │   └── ...
│   ├── logs/                          # Raw per-prompt CSV results
│   │   ├── dvfs_results_Qwen2.5-7B.csv
│   │   ├── dvfs_results_Mistral-7B.csv
│   │   └── dvfs_results_Llama-2-7B.csv
│   └── tables/                        # Aggregated summary tables
│       ├── dvfs_summary_Qwen2.5-7B_default.csv
│       ├── dvfs_summary_Qwen2.5-7B_phase_aware.csv
│       ├── gain_summary_3model.csv
│       └── ...
│
├── results_adaptive_v2/               # Adaptive experiment results
│   ├── figures/
│   ├── logs/
│   └── tables/
│
├── results_llama2/                    # Llama-2-7B results
│   ├── figures/
│   ├── logs/
│   └── tables/
│
└── report/                            # Final report (NeurIPS format)
    └── Phase_Aware_DVFS_Final_Report.pdf
```

## Setup and Installation

### Prerequisites
- Python 3.11+
- NVIDIA GPU with NVML support (tested on A100-SXM4-40GB)
- CUDA 12.1+

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/SreevatsaKolachana/phase-aware-dvfs-llm.git
cd phase-aware-dvfs-llm

# Create conda environment
conda create -n dvfs python=3.11 -y
conda activate dvfs

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
torch>=2.5.1
transformers>=4.43.0
bitsandbytes>=0.43.0
pynvml>=11.5.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
datasets>=2.14.0
scipy>=1.11.0
```

### Download Models

Models should be placed in a `models/` directory (not included in the repo due to size):

```bash
mkdir -p models
cd models

# Qwen2.5-7B-Instruct
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='Qwen2.5-7B-Instruct',
                  ignore_patterns=['*.bin'])
"

# Mistral-7B-Instruct-v0.3
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='Mistral-7B-Instruct-v0.3',
                  ignore_patterns=['*.bin'])
"

# Llama-2-7B-chat (requires Meta license approval)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-2-7b-chat-hf', local_dir='Llama-2-7b-chat-hf',
                  ignore_patterns=['*.bin', 'original/*'])
"
```

### Prepare Dataset

The ShareGPT dataset is cached locally. To regenerate from scratch:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Aeala/ShareGPT_Vicuna_unfiltered')
ds['train'].save_to_disk('data/sharegpt_raw')
"
```

The bucketed prompt files (`sharegpt_bucketed_*.json`) are generated automatically on first run.

## Running Experiments

### On NCSU Hazel HPC (LSF scheduler)

```bash
# 1. Baseline phase-separated benchmark (Qwen2.5-7B)
bsub < src/submit_job.sh

# 2. Static DVFS experiments (Default + Static + Phase-Aware)
bsub < src/submit_dvfs.sh

# 3. Adaptive DVFS experiments (calibrated α sweep)
bsub < src/submit_adaptive_v2.sh

# 4. Cross-architecture (Llama-2-7B)
bsub < src/submit_dvfs_llama2.sh
```

### On any GPU machine

```bash
# Baseline
python src/run_benchmark.py

# Static DVFS
python src/run_dvfs.py

# Adaptive DVFS
python src/run_adaptive_v2.py
```

### Generate Analysis Plots

```bash
# Cross-architecture gain analysis
python analysis/generate_gain_3model.py

# Energy-time correlation plots
python analysis/plot_correlations.py

# Arithmetic intensity computation
python analysis/compute_arithmetic_intensity.py
```

## Method Summary

### Phase Decomposition

Instead of using `model.generate()` (a black box), we call `model()` directly:

```python
# PREFILL — one call, entire prompt (compute-bound, no delay)
outputs = model(input_ids=prompt, use_cache=True)
first_token = argmax(outputs.logits[:, -1, :])
kv_cache = outputs.past_key_values

# DECODE — loop, one token per call (memory-bound, delay inserted)
for step in range(max_new_tokens):
    outputs = model(input_ids=current_token,
                    past_key_values=kv_cache, use_cache=True)
    time.sleep(delay)  # GPU enters idle state, power drops
    next_token = argmax(outputs.logits[:, -1, :])
    kv_cache = outputs.past_key_values
```

### Adaptive Algorithm

```
d_i = clamp(α × (T_budget − t_i), 0, d_max)

Where:
  t_i      = measured compute time for token i (~37ms avg)
  T_budget = 80ms (100ms SLO − 20ms safety margin)
  α        = 0.12 (calibrated aggressiveness)
  d_max    = 10ms (delay cap)
```

### Energy Measurement

GPU power is sampled via NVML at 25ms intervals. Energy is computed by trapezoidal integration of the power trace: `E = ∫ P(t) dt`.

## Hardware Configuration

| Component | Setting |
|-----------|---------|
| GPU | NVIDIA A100-SXM4-40GB |
| SM Clock Range | 210–1410 MHz |
| CPU | AMD EPYC (Hazel compute node) |
| Framework | PyTorch 2.5.1, HuggingFace Transformers |
| Quantization | 4-bit NF4 via bitsandbytes |
| Power Sampling | NVML (pynvml), 25ms intervals |
| Job Scheduler | LSF (IBM Spectrum) |

## Models Evaluated

| Model | d_model | Query Heads | KV Heads | GQA Ratio | d_ff | Layers |
|-------|---------|-------------|----------|-----------|------|--------|
| Qwen2.5-7B | 3584 | 28 | 4 | 7× | 18944 | 28 |
| Mistral-7B | 4096 | 32 | 8 | 4× | 14336 | 32 |
| Llama-2-7B | 4096 | 32 | 32 | 1× (MHA) | 11008 | 32 |

## Expected Output

Each experiment produces:
- **CSV logs** (`results/logs/`): Per-prompt metrics including energy, power, TTFT, TBT, TPS
- **Summary tables** (`results/tables/`): Averaged metrics per (policy, bucket)
- **Figures** (`results/figures/`): Comparison bar charts, throttle sweeps, Pareto plots

## Citation

```bibtex
@article{kolachana2026dvfs,
  title={Phase-Aware DVFS Scheduling for Energy-Efficient Large Language Model Inference},
  author={Kolachana, Sreevatsa},
  journal={ECE 592 Course Project, NC State University},
  year={2026}
}
```

## License

This project is for academic purposes (ECE 592 course project at NC State University).
