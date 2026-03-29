"""
DVFS Simulation for Phase-Aware LLM Inference
==============================================
Since we cannot change GPU clocks on Hazel (no root access),
we simulate DVFS by inserting calibrated delays during decode.

Three policies compared:
  1. DEFAULT:      No throttling (full speed) — already collected as baseline
  2. STATIC:       Same delay on BOTH prefill and decode (simulates fixed power cap)
  3. PHASE-AWARE:  No delay on prefill, delay ONLY on decode (our contribution)

The key insight: during decode, the GPU is memory-bound and has SLO slack.
Adding a small delay between tokens lets the GPU drop to idle power between steps,
reducing total energy while barely affecting user-perceived latency (TBT).
During prefill, we run at full speed to minimize TTFT.
"""

import os
import json
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    "model_name": "Qwen2.5-7B",
    "max_new_tokens": 128,
    "use_4bit": True,
    "samples_per_bucket": 20,  # fewer per bucket since we run 3 policies × 3 buckets
    "hf_token": None,
    "buckets": {
        "short": (1, 64),
        "medium": (65, 256),
        "long": (257, 1024),
    },
    # DVFS simulation parameters
    # These delays simulate running at lower SM frequencies.
    # Calibrated based on A100 decode TBT baseline (~37ms) and
    # GreenLLM's observation that optimal decode freq is ~70% of max.
    # At 70% freq, latency increases by ~1/0.7 = 1.43x, so ~43% more time per token.
    # Baseline TBT ~37ms → simulated TBT ~53ms (within 100ms SLO).
    "throttle_delays_ms": {
        "light":  5.0,    # ~13% slowdown — simulates ~1230 MHz (87% of max)
        "medium": 10.0,   # ~27% slowdown — simulates ~1100 MHz (78% of max)
        "heavy":  20.0,   # ~54% slowdown — simulates ~900 MHz (64% of max)
    },
    # Which delay to use for the main comparison
    "default_throttle": "light",
}


# ============================================================
# GPU POWER SAMPLER (same as baseline)
# ============================================================
class GPUPowerSampler:
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
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
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
# WARMUP
# ============================================================
def warmup_gpu(model, tokenizer, num_warmup=3):
    print(f"Warming up GPU with {num_warmup} passes...")
    warmup_prompt = "Hello, this is a warmup prompt for the GPU."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
    for i in range(num_warmup):
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs.get("attention_mask"), use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            for _ in range(5):
                outputs = model(input_ids=next_token,
                              past_key_values=outputs.past_key_values, use_cache=True)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
    del outputs, next_token, inputs
    torch.cuda.empty_cache()
    print("Warmup complete.")


# ============================================================
# DATASET
# ============================================================
def prepare_dataset(tokenizer, config):
    prompts_file = os.path.join(DATA_DIR, f"sharegpt_bucketed_n{config['samples_per_bucket']}_dvfs.json")

    if os.path.exists(prompts_file):
        print(f"Loading cached prompts from {prompts_file}")
        with open(prompts_file, "r") as f:
            return json.load(f)

    print("Loading ShareGPT dataset from local cache...")
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(DATA_DIR, "sharegpt_raw"))
    print(f"Total conversations: {len(ds)}")

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

    random.seed(42)
    bucketed = {name: [] for name in config["buckets"]}
    for prompt in raw_prompts:
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        for bname, (lo, hi) in config["buckets"].items():
            if lo <= length <= hi:
                bucketed[bname].append(prompt)
                break

    sampled = {}
    n = config["samples_per_bucket"]
    for bname, prompts in bucketed.items():
        sampled[bname] = random.sample(prompts, min(n, len(prompts)))
        lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in sampled[bname]]
        print(f"  {bname}: {len(sampled[bname])} prompts, "
              f"token range [{min(lengths)}, {max(lengths)}], avg={sum(lengths)/len(lengths):.0f}")

    with open(prompts_file, "w") as f:
        json.dump(sampled, f, indent=2)
    return sampled


# ============================================================
# CORE: DVFS-AWARE BENCHMARK
# ============================================================
def run_dvfs_benchmark(model, tokenizer, prompt, bucket, policy,
                       max_new_tokens=128, gpu_index=0,
                       prefill_delay_ms=0.0, decode_delay_ms=0.0):
    """
    Run inference with optional throttling delays to simulate DVFS.
    
    Policies:
      - "default":     prefill_delay=0, decode_delay=0
      - "static":      prefill_delay=D, decode_delay=D  (same delay everywhere)
      - "phase_aware": prefill_delay=0, decode_delay=D  (throttle only decode)
    
    The delay is inserted AFTER each GPU computation step. During the delay,
    the GPU drops to idle/low power, reducing average power and total energy.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_tokens = input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sampler = GPUPowerSampler(gpu_index=gpu_index, interval_ms=25)  # faster sampling for DVFS
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

    # Prefill throttle: only for "static" policy
    if prefill_delay_ms > 0:
        time.sleep(prefill_delay_ms / 1000.0)

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

        # Decode throttle: for "static" and "phase_aware" policies
        if decode_delay_ms > 0:
            time.sleep(decode_delay_ms / 1000.0)

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
        "policy": policy,
        "bucket": bucket,
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "input_tokens": input_tokens,
        "generated_tokens": gen_count,
        "prefill_delay_ms": prefill_delay_ms,
        "decode_delay_ms": decode_delay_ms,
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
        "tbt_trace_ms": (tbt_arr * 1000).tolist(),
    }


# ============================================================
# PLOTTING: DVFS COMPARISON (IMPROVED)
# ============================================================
def generate_dvfs_plots(results_df, model_name):
    """Generate comparison plots across DVFS policies with improved Pareto frontiers."""

    policies = results_df["policy"].unique()
    buckets = ["short", "medium", "long"]
    policy_colors = {"default": "#2196F3", "static": "#FF5722", "phase_aware": "#4CAF50"}
    policy_labels = {"default": "Default", "static": "Static Throttle", "phase_aware": "Phase-Aware"}
    bucket_markers = {"short": "o", "medium": "s", "long": "D"}

    # ── 1. DVFS Policy Comparison (bar charts) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    summary = results_df.groupby(["bucket", "policy"])["ttft_ms"].mean().unstack("policy")
    summary = summary.reindex(buckets)
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors.get(p, "#999") for p in summary.columns])
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("Prefill Latency (TTFT)", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Policy", fontsize=9)

    ax = axes[1]
    summary = results_df.groupby(["bucket", "policy"])["tbt_p95_ms"].mean().unstack("policy")
    summary = summary.reindex(buckets)
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors.get(p, "#999") for p in summary.columns])
    ax.set_ylabel("TBT P95 (ms)", fontsize=11)
    ax.set_title("Decode Latency (TBT P95)", fontsize=12, fontweight="bold")
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100ms SLO")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Policy", fontsize=9)

    ax = axes[2]
    summary = results_df.groupby(["bucket", "policy"])["energy_total_j"].mean().unstack("policy")
    summary = summary.reindex(buckets)
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors.get(p, "#999") for p in summary.columns])
    ax.set_ylabel("Total Energy (J)", fontsize=11)
    ax.set_title("Total Energy Consumption", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Policy", fontsize=9)

    plt.suptitle(f"DVFS Policy Comparison — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_comparison_{model_name}.png"), dpi=150)
    plt.close()

    # ── 2. Energy breakdown: prefill vs decode per policy ──
    fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 5))
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        subset = results_df[results_df["policy"] == policy]
        energy_data = subset.groupby("bucket")[["energy_prefill_j", "energy_decode_j"]].mean()
        energy_data = energy_data.reindex(buckets)
        energy_data.plot(kind="bar", stacked=True, ax=ax, color=["#E91E63", "#3F51B5"])
        ax.set_title(f"{policy_labels.get(policy, policy)}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (J)", fontsize=11)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(["Prefill", "Decode"], fontsize=9)

    plt.suptitle(f"Per-Phase Energy Breakdown — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_energy_breakdown_{model_name}.png"), dpi=150)
    plt.close()

    # ── 3. Avg power comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = results_df.groupby(["bucket", "policy"])["avg_power_w"].mean().unstack("policy")
    summary = summary.reindex(buckets)
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors.get(p, "#999") for p in summary.columns])
    ax.set_ylabel("Average Power (W)", fontsize=11)
    ax.set_title(f"Average GPU Power by Policy — {model_name}", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Policy", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_power_{model_name}.png"), dpi=150)
    plt.close()

    # ── 4. IMPROVED Pareto: Mean points with error bars + frontier line ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- 4a. Energy vs TTFT Pareto ---
    ax = axes[0]
    pareto_points_x = []
    pareto_points_y = []

    for policy in ["default", "static", "phase_aware"]:
        subset = results_df[results_df["policy"] == policy]
        for bucket in buckets:
            bsub = subset[subset["bucket"] == bucket]
            if len(bsub) == 0:
                continue
            mean_ttft = bsub["ttft_ms"].mean()
            mean_energy = bsub["energy_total_j"].mean()
            std_ttft = bsub["ttft_ms"].std()
            std_energy = bsub["energy_total_j"].std()

            ax.errorbar(mean_ttft, mean_energy,
                       xerr=std_ttft, yerr=std_energy,
                       fmt=bucket_markers[bucket], color=policy_colors[policy],
                       markersize=10, capsize=4, capthick=1.5,
                       markeredgecolor="black", markeredgewidth=0.5,
                       label=f"{policy_labels[policy]} ({bucket})" if bucket == "short" else "",
                       alpha=0.85)

            # Annotate each point
            ax.annotate(f"{bucket[0].upper()}", (mean_ttft + 5, mean_energy + 3),
                       fontsize=7, color=policy_colors[policy], fontweight="bold")

            pareto_points_x.append(mean_ttft)
            pareto_points_y.append(mean_energy)

    # Draw Pareto frontier (non-dominated points)
    points = list(zip(pareto_points_x, pareto_points_y))
    points.sort(key=lambda p: p[0])
    frontier_x, frontier_y = [points[0][0]], [points[0][1]]
    min_energy = points[0][1]
    for x, y in points[1:]:
        if y <= min_energy:
            frontier_x.append(x)
            frontier_y.append(y)
            min_energy = y
    ax.plot(frontier_x, frontier_y, "k--", alpha=0.4, linewidth=1.5, label="Pareto Frontier")

    ax.set_xlabel("TTFT (ms)", fontsize=11)
    ax.set_ylabel("Total Energy (J)", fontsize=11)
    ax.set_title("Energy vs Prefill Latency", fontsize=12, fontweight="bold")

    # Custom legend: policies only
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=policy_colors["default"],
               markersize=10, label="Default"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=policy_colors["static"],
               markersize=10, label="Static Throttle"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=policy_colors["phase_aware"],
               markersize=10, label="Phase-Aware"),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.4, label="Pareto Frontier"),
        Line2D([0], [0], marker='o', color='gray', markersize=6, label="Short (o)"),
        Line2D([0], [0], marker='s', color='gray', markersize=6, label="Medium (□)"),
        Line2D([0], [0], marker='D', color='gray', markersize=6, label="Long (◇)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- 4b. Energy vs Throughput Pareto ---
    ax = axes[1]
    pareto_points_x2 = []
    pareto_points_y2 = []

    for policy in ["default", "static", "phase_aware"]:
        subset = results_df[results_df["policy"] == policy]
        for bucket in buckets:
            bsub = subset[subset["bucket"] == bucket]
            if len(bsub) == 0:
                continue
            mean_tps = bsub["tokens_per_sec"].mean()
            mean_energy = bsub["energy_total_j"].mean()
            std_tps = bsub["tokens_per_sec"].std()
            std_energy = bsub["energy_total_j"].std()

            ax.errorbar(mean_tps, mean_energy,
                       xerr=std_tps, yerr=std_energy,
                       fmt=bucket_markers[bucket], color=policy_colors[policy],
                       markersize=10, capsize=4, capthick=1.5,
                       markeredgecolor="black", markeredgewidth=0.5,
                       alpha=0.85)

            ax.annotate(f"{bucket[0].upper()}", (mean_tps + 0.15, mean_energy + 3),
                       fontsize=7, color=policy_colors[policy], fontweight="bold")

            pareto_points_x2.append(mean_tps)
            pareto_points_y2.append(mean_energy)

    # Pareto frontier for throughput (higher TPS + lower energy = better → bottom-right)
    points2 = list(zip(pareto_points_x2, pareto_points_y2))
    points2.sort(key=lambda p: -p[0])  # sort by descending TPS
    frontier_x2, frontier_y2 = [points2[0][0]], [points2[0][1]]
    min_energy2 = points2[0][1]
    for x, y in points2[1:]:
        if y <= min_energy2:
            frontier_x2.append(x)
            frontier_y2.append(y)
            min_energy2 = y
    ax.plot(frontier_x2, frontier_y2, "k--", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Tokens/sec", fontsize=11)
    ax.set_ylabel("Total Energy (J)", fontsize=11)
    ax.set_title("Energy vs Throughput", fontsize=12, fontweight="bold")
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Energy-Performance Pareto Frontier — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_pareto_{model_name}.png"), dpi=150)
    plt.close()

    # ── 5. EDP comparison (bar chart) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = results_df.groupby(["bucket", "policy"])["edp"].mean().unstack("policy")
    summary = summary.reindex(buckets)
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors.get(p, "#999") for p in summary.columns])
    ax.set_ylabel("EDP (J·s)", fontsize=11)
    ax.set_title(f"Energy-Delay Product by Policy — {model_name}", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Policy", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_edp_{model_name}.png"), dpi=150)
    plt.close()

    # ── 6. Throttle sweep with improved styling ──
    if "throttle_level" in results_df.columns:
        sweep_df = results_df[results_df["policy"] == "phase_aware"]
        if len(sweep_df) > 0:
            sweep_summary = sweep_df.groupby("decode_delay_ms").agg({
                "energy_total_j": ["mean", "std"],
                "tbt_p95_ms": ["mean", "std"],
                "tokens_per_sec": ["mean", "std"],
                "avg_power_w": ["mean", "std"],
            }).reset_index()
            sweep_summary.columns = ['_'.join(col).strip('_') for col in sweep_summary.columns]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Energy vs delay
            ax = axes[0, 0]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["energy_total_j_mean"],
                       yerr=sweep_summary["energy_total_j_std"], fmt="o-", color="#4CAF50",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            # Mark the minimum
            min_idx = sweep_summary["energy_total_j_mean"].idxmin()
            ax.scatter(sweep_summary.loc[min_idx, "decode_delay_ms"],
                      sweep_summary.loc[min_idx, "energy_total_j_mean"],
                      s=200, facecolors="none", edgecolors="red", linewidths=2.5, zorder=5)
            ax.annotate("Optimal", (sweep_summary.loc[min_idx, "decode_delay_ms"],
                        sweep_summary.loc[min_idx, "energy_total_j_mean"]),
                       textcoords="offset points", xytext=(15, 10), fontsize=10,
                       color="red", fontweight="bold")
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("Total Energy (J)", fontsize=11)
            ax.set_title("Energy vs Decode Throttle", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # TBT vs delay
            ax = axes[0, 1]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["tbt_p95_ms_mean"],
                       yerr=sweep_summary["tbt_p95_ms_std"], fmt="o-", color="#E91E63",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="100ms SLO")
            ax.fill_between([0, 25], 100, 120, color="red", alpha=0.05)
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("TBT P95 (ms)", fontsize=11)
            ax.set_title("Decode Latency vs Throttle", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Power vs delay
            ax = axes[1, 0]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["avg_power_w_mean"],
                       yerr=sweep_summary["avg_power_w_std"], fmt="s-", color="#FF9800",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("Average Power (W)", fontsize=11)
            ax.set_title("GPU Power vs Throttle", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # TPS vs delay
            ax = axes[1, 1]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["tokens_per_sec_mean"],
                       yerr=sweep_summary["tokens_per_sec_std"], fmt="D-", color="#9C27B0",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("Tokens/sec", fontsize=11)
            ax.set_title("Throughput vs Throttle", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            plt.suptitle(f"Throttle Sweep Analysis — {model_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f"dvfs_sweep_{model_name}.png"), dpi=150)
            plt.close()

    print(f"DVFS plots saved to {FIGURES_DIR}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("DVFS Simulation Benchmark")
    print("=" * 60)

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

    hf_token = os.environ.get("HF_TOKEN", CONFIG["hf_token"])
    model_name = CONFIG["model_name"]

    # Load model
    tokenizer, model = load_model_and_tokenizer(CONFIG["model_id"], CONFIG["use_4bit"], hf_token)

    # Prepare dataset
    prompts = prepare_dataset(tokenizer, CONFIG)

    # Warmup
    warmup_gpu(model, tokenizer)

    # Get the default throttle delay
    throttle_key = CONFIG["default_throttle"]
    delay_ms = CONFIG["throttle_delays_ms"][throttle_key]
    print(f"\nThrottle level: {throttle_key} ({delay_ms}ms decode delay)")

    # ── Define the three policies ──
    policies = {
        "default":     {"prefill_delay_ms": 0.0,      "decode_delay_ms": 0.0},
        "static":      {"prefill_delay_ms": delay_ms,  "decode_delay_ms": delay_ms},
        "phase_aware": {"prefill_delay_ms": 0.0,       "decode_delay_ms": delay_ms},
    }

    # ── Run all policies ──
    all_results = []
    total_runs = len(policies) * sum(len(v) for v in prompts.values())
    run_idx = 0

    for policy_name, delays in policies.items():
        print(f"\n{'#'*60}")
        print(f"# POLICY: {policy_name}")
        print(f"# Prefill delay: {delays['prefill_delay_ms']}ms")
        print(f"# Decode delay:  {delays['decode_delay_ms']}ms")
        print(f"{'#'*60}")

        # Shuffle prompts for this policy
        tasks = []
        for bucket, prompt_list in prompts.items():
            for prompt in prompt_list:
                tasks.append((bucket, prompt))
        random.seed(42)
        random.shuffle(tasks)

        for bucket, prompt in tasks:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {policy_name:12s} {bucket:6s} | ", end="", flush=True)

            result = run_dvfs_benchmark(
                model, tokenizer, prompt, bucket, policy_name,
                max_new_tokens=CONFIG["max_new_tokens"],
                prefill_delay_ms=delays["prefill_delay_ms"],
                decode_delay_ms=delays["decode_delay_ms"],
            )
            result["throttle_level"] = throttle_key

            print(f"TTFT={result['ttft_ms']:.1f}ms | "
                  f"TPS={result['tokens_per_sec']:.1f} | "
                  f"Power={result['avg_power_w']:.1f}W | "
                  f"Energy={result['energy_total_j']:.1f}J")

            all_results.append(result)
            time.sleep(0.3)

    results_df = pd.DataFrame(all_results)

    # ── Also run a throttle sweep with phase_aware at different levels ──
    print(f"\n{'#'*60}")
    print(f"# THROTTLE SWEEP (phase_aware at all delay levels)")
    print(f"{'#'*60}")

    # Use just medium bucket for the sweep (representative)
    sweep_prompts = prompts.get("medium", prompts.get("short", []))[:10]

    for level_name, level_delay in CONFIG["throttle_delays_ms"].items():
        print(f"\n  Sweep: {level_name} ({level_delay}ms)")
        for idx, prompt in enumerate(sweep_prompts, 1):
            print(f"    [{idx}/{len(sweep_prompts)}] ", end="", flush=True)
            result = run_dvfs_benchmark(
                model, tokenizer, prompt, "medium", "phase_aware",
                max_new_tokens=CONFIG["max_new_tokens"],
                prefill_delay_ms=0.0,
                decode_delay_ms=level_delay,
            )
            result["throttle_level"] = level_name
            print(f"delay={level_delay}ms | TPS={result['tokens_per_sec']:.1f} | "
                  f"Power={result['avg_power_w']:.1f}W | Energy={result['energy_total_j']:.1f}J")
            all_results.append(result)
            time.sleep(0.2)

    results_df = pd.DataFrame(all_results)

    # ── Save results ──
    save_df = results_df.drop(columns=["tbt_trace_ms"], errors="ignore")
    csv_path = os.path.join(LOGS_DIR, f"dvfs_results_{model_name}.csv")
    save_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary per policy ──
    summary_cols = [
        "input_tokens", "generated_tokens", "ttft_ms", "tbt_p95_ms",
        "tokens_per_sec", "ms_per_token",
        "energy_prefill_j", "energy_decode_j", "energy_total_j",
        "energy_per_token_mj", "avg_power_w", "edp"
    ]

    # Filter to main comparison (not sweep)
    main_df = results_df[results_df["throttle_level"] == throttle_key]

    for policy in ["default", "static", "phase_aware"]:
        pdf = main_df[main_df["policy"] == policy]
        if len(pdf) == 0:
            continue
        summary = pdf.groupby("bucket")[summary_cols].mean().round(3)
        print(f"\n{'='*80}")
        print(f"DVFS SUMMARY — {model_name} — Policy: {policy}")
        print(f"{'='*80}")
        print(summary.to_string())

    # ── Compute savings ──
    print(f"\n{'='*80}")
    print(f"ENERGY SAVINGS COMPARISON")
    print(f"{'='*80}")

    for bucket in ["short", "medium", "long"]:
        default_energy = main_df[(main_df["policy"] == "default") & (main_df["bucket"] == bucket)]["energy_total_j"].mean()
        static_energy = main_df[(main_df["policy"] == "static") & (main_df["bucket"] == bucket)]["energy_total_j"].mean()
        phase_energy = main_df[(main_df["policy"] == "phase_aware") & (main_df["bucket"] == bucket)]["energy_total_j"].mean()

        default_ttft = main_df[(main_df["policy"] == "default") & (main_df["bucket"] == bucket)]["ttft_ms"].mean()
        static_ttft = main_df[(main_df["policy"] == "static") & (main_df["bucket"] == bucket)]["ttft_ms"].mean()
        phase_ttft = main_df[(main_df["policy"] == "phase_aware") & (main_df["bucket"] == bucket)]["ttft_ms"].mean()

        static_saving = (1 - static_energy / default_energy) * 100
        phase_saving = (1 - phase_energy / default_energy) * 100
        static_ttft_penalty = (static_ttft / default_ttft - 1) * 100
        phase_ttft_penalty = (phase_ttft / default_ttft - 1) * 100

        print(f"\n  {bucket:8s}:")
        print(f"    Default:     Energy={default_energy:.1f}J  TTFT={default_ttft:.1f}ms")
        print(f"    Static:      Energy={static_energy:.1f}J ({static_saving:+.1f}%)  TTFT={static_ttft:.1f}ms ({static_ttft_penalty:+.1f}%)")
        print(f"    Phase-Aware: Energy={phase_energy:.1f}J ({phase_saving:+.1f}%)  TTFT={phase_ttft:.1f}ms ({phase_ttft_penalty:+.1f}%)")

    print(f"\n{'='*80}")

    # Save summaries
    for policy in ["default", "static", "phase_aware"]:
        pdf = main_df[main_df["policy"] == policy]
        if len(pdf) > 0:
            summary = pdf.groupby("bucket")[summary_cols].mean().round(3)
            summary.to_csv(os.path.join(TABLES_DIR, f"dvfs_summary_{model_name}_{policy}.csv"))

    # ── Plots ──
    generate_dvfs_plots(results_df, model_name)

    print("\nDVFS simulation complete!")


if __name__ == "__main__":
    main()

