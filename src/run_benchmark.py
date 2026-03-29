"""
Phase-Aware LLM Inference Benchmark for NCSU Hazel HPC
=======================================================
Runs on A100 GPUs via LSF job scheduler.
Measures prefill/decode phases separately with continuous power sampling.
"""

import os
import json
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC (no display)
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pynvml

# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIG = {
    "model_id": os.path.join(PROJECT_DIR, "models/Qwen2.5-7B-Instruct"),
    "max_new_tokens": 128,
    "use_4bit": True,
    "samples_per_bucket": 40,
    "hf_token": None,  # Set via environment variable HF_TOKEN
    "buckets": {
        "short": (1, 64),
        "medium": (65, 256),
        "long": (257, 1024),
    },
}


# ============================================================
# GPU POWER SAMPLER
# ============================================================
class GPUPowerSampler:
    """Background thread polling GPU power via NVML every interval_ms."""

    def __init__(self, gpu_index=0, interval_ms=50):
        self.gpu_index = gpu_index
        self.interval_s = interval_ms / 1000.0
        self._samples = []
        self._running = False
        self._thread = None
        self._handle = None

    def _poll_loop(self):
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.perf_counter(), power_mw / 1000.0))
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval_s)

    def start(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def get_power_trace(self):
        return list(self._samples)

    def get_energy_joules(self):
        if len(self._samples) < 2:
            return 0.0
        ts = [s[0] for s in self._samples]
        pw = [s[1] for s in self._samples]
        return float(np.trapz(pw, ts))

    def get_avg_power_watts(self):
        if not self._samples:
            return 0.0
        return float(np.mean([s[1] for s in self._samples]))

    def get_energy_in_range(self, t_start, t_end):
        subset = [(t, p) for t, p in self._samples if t_start <= t <= t_end]
        if len(subset) < 2:
            return self.get_avg_power_watts() * (t_end - t_start)
        return float(np.trapz([s[1] for s in subset], [s[0] for s in subset]))


# ============================================================
# DATASET PREPARATION
# ============================================================
def prepare_dataset(tokenizer, config):
    """Download ShareGPT, extract first user turns, bucket by token length."""
    from datasets import load_dataset

    prompts_file = os.path.join(DATA_DIR, f"sharegpt_bucketed_n{config['samples_per_bucket']}.json")

    # Check if already prepared with same sample count
    if os.path.exists(prompts_file):
        print(f"Loading cached prompts from {prompts_file}")
        with open(prompts_file, "r") as f:
            return json.load(f)

    print("Loading ShareGPT dataset from local cache...")
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(DATA_DIR, "sharegpt_raw"))
    print(f"Total conversations: {len(ds)}")

    # Extract first user message
    raw_prompts = []
    for row in ds:
        convos = row.get("conversations", [])
        if convos is None:
            continue
        for turn in convos:
            if isinstance(turn, dict):
                role = turn.get("from", turn.get("role", ""))
                text = turn.get("value", turn.get("content", ""))
                if role == "human" and isinstance(text, str) and len(text.strip()) > 10:
                    raw_prompts.append(text.strip())
                    break
    print(f"Extracted {len(raw_prompts)} first-turn user prompts")

    # Tokenize and bucket
    random.seed(42)
    bucketed = {name: [] for name in config["buckets"]}

    for prompt in raw_prompts:
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        for bname, (lo, hi) in config["buckets"].items():
            if lo <= length <= hi:
                bucketed[bname].append(prompt)
                break

    # Sample
    sampled = {}
    n = config["samples_per_bucket"]
    for bname, prompts in bucketed.items():
        if len(prompts) >= n:
            sampled[bname] = random.sample(prompts, n)
        else:
            sampled[bname] = prompts
        lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in sampled[bname]]
        print(f"  {bname}: {len(sampled[bname])} prompts, "
              f"token range [{min(lengths)}, {max(lengths)}], avg={sum(lengths)/len(lengths):.0f}")

    # Cache to disk
    with open(prompts_file, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved bucketed prompts to {prompts_file}")

    return sampled


# ============================================================
# MODEL LOADING
# ============================================================
def load_model_and_tokenizer(model_id, use_4bit=True, hf_token=None):
    print(f"Loading model: {model_id} (4bit={use_4bit})")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16, token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, token=hf_token,
        )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s | Device: {model.device}")
    return tokenizer, model


# ============================================================
# PHASE-AWARE BENCHMARK
# ============================================================
def run_phase_aware_benchmark(model, tokenizer, prompt, bucket,
                               max_new_tokens=128, gpu_index=0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_tokens = input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sampler = GPUPowerSampler(gpu_index=gpu_index, interval_ms=50)
    sampler.start()

    generated_token_ids = []
    tbt_list = []

    # ---- PREFILL ----
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=True,
        )

    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()

    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated_token_ids.append(next_token_id.item())
    ttft = t_prefill_end - t_prefill_start

    # ---- DECODE ----
    current_token = next_token_id
    t_decode_start = time.perf_counter()
    t_prev = t_decode_start

    for step in range(max_new_tokens - 1):
        torch.cuda.synchronize()

        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t_step_end = time.perf_counter()

        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        token_id = next_token_id.item()
        generated_token_ids.append(token_id)

        tbt_list.append(t_step_end - t_prev)
        t_prev = t_step_end

        if token_id == tokenizer.eos_token_id:
            break
        current_token = next_token_id

    t_decode_end = time.perf_counter()
    sampler.stop()

    # ---- METRICS ----
    total_time = t_decode_end - t_prefill_start
    decode_time = t_decode_end - t_decode_start
    gen_count = len(generated_token_ids)

    energy_prefill = sampler.get_energy_in_range(t_prefill_start, t_prefill_end)
    energy_decode = sampler.get_energy_in_range(t_decode_start, t_decode_end)
    energy_total = sampler.get_energy_in_range(t_prefill_start, t_decode_end)
    avg_power = sampler.get_avg_power_watts()

    tbt_arr = np.array(tbt_list) if tbt_list else np.array([0.0])
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    return {
        "bucket": bucket,
        "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "input_tokens": input_tokens,
        "generated_tokens": gen_count,
        "ttft_ms": ttft * 1000,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "tokens_per_sec": gen_count / max(total_time, 1e-8),
        "ms_per_token": (total_time / max(gen_count, 1)) * 1000,
        "tbt_mean_ms": float(np.mean(tbt_arr) * 1000),
        "tbt_p50_ms": float(np.percentile(tbt_arr, 50) * 1000),
        "tbt_p90_ms": float(np.percentile(tbt_arr, 90) * 1000),
        "tbt_p95_ms": float(np.percentile(tbt_arr, 95) * 1000),
        "energy_prefill_j": energy_prefill,
        "energy_decode_j": energy_decode,
        "energy_total_j": energy_total,
        "energy_per_token_mj": (energy_total / max(gen_count, 1)) * 1000,
        "avg_power_w": avg_power,
        "edp": energy_total * total_time,
        "peak_mem_mb": peak_mem_mb,
        "output_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "tbt_trace_ms": (tbt_arr * 1000).tolist(),
    }


# ============================================================
# PLOTTING
# ============================================================
def generate_plots(results_df):
    """Generate all plots and save to figures directory."""

    # 1. Overview: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    results_df.groupby("bucket")["ttft_ms"].mean().plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set_ylabel("TTFT (ms)"); ax.set_title("Prefill Latency"); ax.tick_params(axis='x', rotation=0)

    ax = axes[0, 1]
    results_df.groupby("bucket")[["energy_prefill_j", "energy_decode_j"]].mean().plot(
        kind="bar", stacked=True, ax=ax, color=["#E91E63", "#3F51B5"])
    ax.set_ylabel("Energy (J)"); ax.set_title("Energy: Prefill vs Decode"); ax.tick_params(axis='x', rotation=0)

    ax = axes[1, 0]
    results_df.groupby("bucket")["energy_per_token_mj"].mean().plot(kind="bar", ax=ax, color="#9C27B0")
    ax.set_ylabel("mJ/token"); ax.set_title("Energy per Token"); ax.tick_params(axis='x', rotation=0)

    ax = axes[1, 1]
    results_df.groupby("bucket")["edp"].mean().plot(kind="bar", ax=ax, color="#607D8B")
    ax.set_ylabel("J·s"); ax.set_title("Energy-Delay Product"); ax.tick_params(axis='x', rotation=0)

    plt.suptitle("Phase-Aware Inference Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "phase_metrics_overview.png"), dpi=150)
    plt.close()

    # 2. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(results_df["input_tokens"], results_df["ttft_ms"],
               c="#E91E63", alpha=0.7, edgecolors="black", s=60)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("TTFT (ms)"); ax.set_title("Prompt Length vs Prefill Latency")

    ax = axes[1]
    ax.scatter(results_df["input_tokens"], results_df["energy_per_token_mj"],
               c="#3F51B5", alpha=0.7, edgecolors="black", s=60)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("mJ/token"); ax.set_title("Prompt Length vs Energy Efficiency")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prompt_length_analysis.png"), dpi=150)
    plt.close()

    # 3. TBT distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"short": "#4CAF50", "medium": "#FF9800", "long": "#E91E63"}
    for bucket in results_df["bucket"].unique():
        subset = results_df[results_df["bucket"] == bucket]
        all_tbts = []
        for trace in subset["tbt_trace_ms"]:
            all_tbts.extend(trace)
        if all_tbts:
            ax.hist(all_tbts, bins=30, alpha=0.5, label=bucket, color=colors.get(bucket, "#999"))
    ax.set_xlabel("Time Between Tokens (ms)"); ax.set_ylabel("Count")
    ax.set_title("TBT Distribution by Prompt Bucket"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tbt_distribution.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {FIGURES_DIR}")


# ============================================================
# WARMUP
# ============================================================
def warmup_gpu(model, tokenizer, num_warmup=3):
    """
    Run a few throwaway inference passes to warm up the GPU.
    This eliminates first-run overhead (CUDA kernel compilation,
    memory allocation) that would skew the first bucket's results.
    """
    print(f"Warming up GPU with {num_warmup} passes...")
    warmup_prompt = "Hello, this is a warmup prompt for the GPU."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)

    for i in range(num_warmup):
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
            )
            # Also do a few decode steps to warm up that path
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            for _ in range(5):
                outputs = model(
                    input_ids=next_token,
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()

    # Clear any memory from warmup
    del outputs, next_token, inputs
    torch.cuda.empty_cache()
    print("Warmup complete.")


# ============================================================
# MAIN
# ============================================================
# List of models to benchmark. Each entry: (display_name, local_path)
# Add more models here — just download them on the login node first.
MODELS = [
    ("Qwen2.5-7B", os.path.join(PROJECT_DIR, "models/Qwen2.5-7B-Instruct")),
    # Uncomment after downloading on login node:
    # ("Llama3.1-8B", os.path.join(PROJECT_DIR, "models/Llama-3.1-8B-Instruct")),
    ("Mistral-7B", os.path.join(PROJECT_DIR, "models/Mistral-7B-Instruct-v0.3")),
]


def main():
    print("=" * 60)
    print("Phase-Aware LLM Inference Benchmark")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # NVML check
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        print(f"NVML power read OK: {power_mw / 1000.0:.1f} W")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"NVML power read FAILED: {e}")

    # HF token from environment
    hf_token = os.environ.get("HF_TOKEN", CONFIG["hf_token"])

    # Run benchmark for each model
    for model_name, model_path in MODELS:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name}")
        print(f"# Path:  {model_path}")
        print(f"{'#'*60}")

        if not os.path.exists(model_path):
            print(f"SKIPPING — model not found at {model_path}")
            print(f"Download it on the login node first.")
            continue

        # Load model
        tokenizer, model = load_model_and_tokenizer(model_path, CONFIG["use_4bit"], hf_token)

        # Prepare dataset (uses tokenizer for length measurement)
        prompts = prepare_dataset(tokenizer, CONFIG)

        # Warmup GPU to eliminate first-run overhead
        warmup_gpu(model, tokenizer)

        # Interleave buckets to avoid ordering bias
        # Instead of running all short, then all medium, then all long,
        # we shuffle the full list of (bucket, prompt) pairs
        all_tasks = []
        for bucket, prompt_list in prompts.items():
            for prompt in prompt_list:
                all_tasks.append((bucket, prompt))
        random.seed(42)
        random.shuffle(all_tasks)

        print(f"\nRunning {len(all_tasks)} prompts (shuffled across buckets)...")

        all_results = []
        for idx, (bucket, prompt) in enumerate(all_tasks, 1):
            print(f"  [{idx}/{len(all_tasks)}] {bucket:6s} | ", end="", flush=True)
            result = run_phase_aware_benchmark(
                model, tokenizer, prompt, bucket,
                max_new_tokens=CONFIG["max_new_tokens"])
            result["model"] = model_name  # tag results with model name
            print(f"TTFT={result['ttft_ms']:.1f}ms | "
                  f"TPS={result['tokens_per_sec']:.1f} | "
                  f"Energy={result['energy_total_j']:.2f}J | "
                  f"EDP={result['edp']:.4f}")
            all_results.append(result)
            time.sleep(0.3)

        results_df = pd.DataFrame(all_results)

        # Save raw results
        save_df = results_df.drop(columns=["tbt_trace_ms", "output_text"], errors="ignore")
        csv_path = os.path.join(LOGS_DIR, f"run_logs_{model_name}.csv")
        save_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        # Summary
        summary_cols = [
            "input_tokens", "generated_tokens", "ttft_ms", "tbt_p95_ms",
            "tokens_per_sec", "ms_per_token",
            "energy_prefill_j", "energy_decode_j", "energy_total_j",
            "energy_per_token_mj", "avg_power_w", "edp", "peak_mem_mb"
        ]
        summary = results_df.groupby("bucket")[summary_cols].mean().round(3)
        print(f"\n{'='*80}")
        print(f"PHASE-AWARE INFERENCE SUMMARY — {model_name}")
        print(f"{'='*80}")
        print(summary.to_string())
        print(f"{'='*80}")

        summary_path = os.path.join(TABLES_DIR, f"phase_summary_{model_name}.csv")
        summary.to_csv(summary_path)
        print(f"Saved: {summary_path}")

        # Plots
        generate_plots(results_df)

        # Free GPU memory before next model
        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    print("\nAll models complete!")


if __name__ == "__main__":
    main()

