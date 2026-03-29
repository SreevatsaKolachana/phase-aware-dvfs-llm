# Phase-Aware DVFS Scheduling for Energy-Efficient LLM Inference

**ECE 592: Generative AI/ML — NC State University**  
**Author:** Sreevatsa Kolachana (svkolach)

## Overview

This project investigates phase-aware dynamic voltage and frequency scaling (DVFS) for energy-efficient LLM inference on GPUs. LLM inference has two distinct phases — compute-bound **prefill** and memory-bound **decode** — with different power characteristics. We demonstrate that throttling only the decode phase (phase-aware scheduling) achieves energy savings comparable to static throttling while preserving prefill latency (TTFT).

## Key Results

- **3–5% energy savings** with phase-aware decode throttling on Qwen2.5-7B
- **Phase-aware preserves TTFT** (< 3% penalty) vs static throttle (+3% penalty)
- **U-shaped energy curve** validated: optimal throttle at 5ms decode delay
- All TBT values remain well within 100ms SLO constraint
- Results validated on two models: Qwen2.5-7B-Instruct and Mistral-7B-Instruct

## Code Organization
src/
├── run_benchmark.py     # Baseline phase-aware benchmark (TTFT, TBT, energy)
├── run_dvfs.py          # DVFS simulation (3 policies + throttle sweep)
├── submit_job.sh        # LSF job script for baseline benchmark
└── submit_dvfs.sh       # LSF job script for DVFS simulation
results/
├── figures/             # All plots (comparison, Pareto, sweep, EDP)
├── tables/              # Summary CSVs per model per policy
└── logs/                # Raw per-prompt measurement CSVs
data/
├── sharegpt_bucketed_n40.json    # 40 prompts/bucket for baseline
└── sharegpt_bucketed_n20_dvfs.json  # 20 prompts/bucket for DVFS

## Experimental Setup

- **Hardware:** NVIDIA A100-SXM4-40GB (NCSU Hazel HPC cluster)
- **Models:** Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 (4-bit quantized)
- **Dataset:** ShareGPT (real ChatGPT conversations), bucketed by prompt length
  - Short: 1–64 tokens | Medium: 65–256 tokens | Long: 257–1024 tokens
- **Metrics:** TTFT, TBT (P95), tokens/sec, energy (J), EDP, average power (W)

## Running Instructions

### Prerequisites
```bash
conda create -n dvfs python=3.11 -y
conda activate dvfs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes pynvml pandas matplotlib numpy datasets
```

### Download models (on login node with internet)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='./models/Qwen2.5-7B-Instruct')
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='./models/Mistral-7B-Instruct-v0.3')
"
```

### Run baseline benchmark
```bash
bsub < src/submit_job.sh
```

### Run DVFS simulation
```bash
bsub < src/submit_dvfs.sh
```

## References

- GreenLLM (Liu et al., 2025): SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving
- Kachris (2025): A Survey on Hardware Accelerators for Large Language Models
