"""
Adaptive Phase-Aware DVFS v2 — Calibrated Parameters
=====================================================
v1 used alpha=0.5 which produced ~15ms average delay — way past the 
optimal 5ms point on the U-shaped energy curve.

Fix: The U-curve showed optimal at 5ms delay on ~37ms baseline TBT.
Slack = T_budget - t_i ≈ 80 - 37 = 43ms.
To get 5ms average delay: alpha = 5/43 ≈ 0.12

This version sweeps alpha in the correct range: {0.05, 0.10, 0.15, 0.20}
and caps d_max at 10ms (since we know >10ms is past the U-curve optimum).

The key insight: adaptive should match or slightly exceed the static 5ms 
average delay, but DISTRIBUTE that delay intelligently — more delay on 
fast tokens, less on slow tokens.
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
from matplotlib.lines import Line2D
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pynvml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results_adaptive_v2")
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
    "samples_per_bucket": 20,
    "hf_token": None,
    "buckets": {
        "short": (1, 64),
        "medium": (65, 256),
        "long": (257, 1024),
    },
    # Static (previous scheme)
    "static_decode_delay_ms": 5.0,
    # Adaptive params — CALIBRATED
    "T_SLO_ms": 100.0,
    "T_margin_ms": 20.0,       # T_budget = 80ms
    "d_max_ms": 10.0,          # Cap at 10ms (past this is bad per U-curve)
    "default_alpha": 0.12,     # ~5ms avg delay on typical 37ms tokens
    "alpha_sweep": [0.05, 0.10, 0.12, 0.15, 0.20],
}

T_BUDGET_MS = CONFIG["T_SLO_ms"] - CONFIG["T_margin_ms"]


class GPUPowerSampler:
    def __init__(self, gpu_index=0, interval_ms=25):
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

    def get_energy_in_range(self, t_start, t_end):
        subset = [(t, p) for t, p in self._samples if t_start <= t <= t_end]
        if len(subset) < 2:
            return self.get_avg_power_watts() * (t_end - t_start)
        return float(np.trapz([s[1] for s in subset], [s[0] for s in subset]))

    def get_avg_power_watts(self):
        if not self._samples:
            return 0.0
        return float(np.mean([s[1] for s in self._samples]))


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
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return tokenizer, model


def warmup_gpu(model, tokenizer, num_warmup=3):
    print(f"Warming up GPU with {num_warmup} passes...")
    warmup_prompt = "Hello, this is a warmup prompt for the GPU."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
    for _ in range(num_warmup):
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


def prepare_dataset(tokenizer, config):
    prompts_file = os.path.join(DATA_DIR, f"sharegpt_bucketed_n{config['samples_per_bucket']}_dvfs.json")
    if os.path.exists(prompts_file):
        print(f"Loading cached prompts from {prompts_file}")
        with open(prompts_file, "r") as f:
            return json.load(f)
    print("ERROR: prompts file not found. Run the original run_dvfs.py first.")
    return {}


def run_benchmark(model, tokenizer, prompt, bucket, policy,
                  max_new_tokens=128, gpu_index=0,
                  static_delay_ms=0.0,
                  alpha=0.12, T_budget_ms=80.0, d_max_ms=10.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    input_tokens = input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sampler = GPUPowerSampler(gpu_index=gpu_index, interval_ms=25)
    sampler.start()

    generated_token_ids = []
    tbt_list = []
    delay_trace = []
    compute_trace = []
    slack_trace = []

    # PREFILL
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                       attention_mask=inputs.get("attention_mask"), use_cache=True)
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()

    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated_token_ids.append(next_token_id.item())
    ttft = t_prefill_end - t_prefill_start

    # DECODE
    current_token = next_token_id
    t_decode_start = time.perf_counter()
    t_prev = t_decode_start

    for step in range(max_new_tokens - 1):
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(input_ids=current_token,
                           past_key_values=past_key_values, use_cache=True)

        torch.cuda.synchronize()
        t_compute_end = time.perf_counter()
        t_i_ms = (t_compute_end - t_step_start) * 1000.0

        # DELAY DECISION
        if policy == "default":
            d_i_ms = 0.0
            slack_ms = T_budget_ms - t_i_ms
        elif policy == "static":
            d_i_ms = static_delay_ms
            slack_ms = T_budget_ms - t_i_ms
        elif policy == "adaptive":
            slack_ms = T_budget_ms - t_i_ms
            if slack_ms > 0:
                d_i_ms = min(alpha * slack_ms, d_max_ms)
            else:
                d_i_ms = 0.0
        else:
            d_i_ms = 0.0
            slack_ms = 0.0

        if d_i_ms > 0:
            time.sleep(d_i_ms / 1000.0)

        t_step_end = time.perf_counter()

        compute_trace.append(t_i_ms)
        delay_trace.append(d_i_ms)
        slack_trace.append(slack_ms)

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

    total_time = t_decode_end - t_prefill_start
    decode_time = t_decode_end - t_decode_start
    gen_count = len(generated_token_ids)
    energy_prefill = sampler.get_energy_in_range(t_prefill_start, t_prefill_end)
    energy_decode = sampler.get_energy_in_range(t_decode_start, t_decode_end)
    energy_total = sampler.get_energy_in_range(t_prefill_start, t_decode_end)
    avg_power = sampler.get_avg_power_watts()
    tbt_arr = np.array(tbt_list) if tbt_list else np.array([0.0])
    delay_arr = np.array(delay_trace) if delay_trace else np.array([0.0])
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    return {
        "policy": policy, "bucket": bucket,
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "input_tokens": input_tokens, "generated_tokens": gen_count,
        "alpha": alpha if policy == "adaptive" else None,
        "ttft_ms": ttft * 1000, "decode_time_s": decode_time, "total_time_s": total_time,
        "tokens_per_sec": gen_count / max(total_time, 1e-8),
        "ms_per_token": (total_time / max(gen_count, 1)) * 1000,
        "tbt_mean_ms": float(np.mean(tbt_arr) * 1000),
        "tbt_p50_ms": float(np.percentile(tbt_arr, 50) * 1000),
        "tbt_p90_ms": float(np.percentile(tbt_arr, 90) * 1000),
        "tbt_p95_ms": float(np.percentile(tbt_arr, 95) * 1000),
        "tbt_max_ms": float(np.max(tbt_arr) * 1000),
        "energy_prefill_j": energy_prefill, "energy_decode_j": energy_decode,
        "energy_total_j": energy_total,
        "energy_per_token_mj": (energy_total / max(gen_count, 1)) * 1000,
        "avg_power_w": avg_power, "edp": energy_total * total_time,
        "peak_mem_mb": peak_mem_mb,
        "delay_mean_ms": float(np.mean(delay_arr)),
        "delay_max_ms": float(np.max(delay_arr)),
        "delay_min_ms": float(np.min(delay_arr)),
        "delay_std_ms": float(np.std(delay_arr)),
        "compute_mean_ms": float(np.mean(np.array(compute_trace))) if compute_trace else 0.0,
        "slack_mean_ms": float(np.mean(np.array(slack_trace))) if slack_trace else 0.0,
        "slo_violations": int(np.sum(tbt_arr * 1000 > CONFIG["T_SLO_ms"])),
        "tbt_trace_ms": (tbt_arr * 1000).tolist(),
        "delay_trace_ms": delay_arr.tolist(),
    }


def generate_plots(results_df, model_name):
    buckets = ["short", "medium", "long"]
    main_policies = ["default", "static", "adaptive_0.12"]
    policy_colors = {"default": "#2196F3", "static": "#FF5722", "adaptive_0.12": "#4CAF50"}
    policy_labels = {"default": "Default", "static": "Static (5ms)", "adaptive_0.12": "Adaptive α=0.12"}

    # 1. Main comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (metric, ylabel, title) in enumerate([
        ("energy_total_j", "Total Energy (J)", "Energy Consumption"),
        ("ttft_ms", "TTFT (ms)", "Prefill Latency"),
        ("tbt_p95_ms", "TBT P95 (ms)", "Decode Latency"),
    ]):
        ax = axes[idx]
        summary = results_df[results_df["policy"].isin(main_policies)]
        summary = summary.groupby(["bucket", "policy"])[metric].mean().unstack("policy")
        summary = summary.reindex(buckets)[main_policies]
        summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
            color=[policy_colors[p] for p in main_policies])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=0)
        ax.legend(title="Policy", fontsize=9)
        if metric == "tbt_p95_ms":
            ax.axhline(y=100, color="red", linestyle="--", alpha=0.5)
    plt.suptitle(f"Default vs Static vs Adaptive (calibrated) — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_v2_comparison.png"), dpi=150)
    plt.close()

    # 2. Energy savings
    fig, ax = plt.subplots(figsize=(10, 5))
    savings_data = []
    for bucket in buckets:
        default_e = results_df[(results_df["policy"]=="default") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
        for pol, label in [("static", "Static (5ms)"), ("adaptive_0.12", "Adaptive α=0.12")]:
            pol_e = results_df[(results_df["policy"]==pol) & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
            savings_data.append({"bucket": bucket, "policy": label, "saving_pct": (1-pol_e/default_e)*100})
    sdf = pd.DataFrame(savings_data).pivot(index="bucket", columns="policy", values="saving_pct").reindex(buckets)
    sdf.plot(kind="bar", ax=ax, color=["#4CAF50", "#FF5722"], edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Energy Saving vs Default (%)", fontsize=12)
    ax.set_title(f"Energy Savings Comparison — {model_name}", fontsize=13, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_v2_savings.png"), dpi=150)
    plt.close()

    # 3. Power comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = results_df[results_df["policy"].isin(main_policies)]
    summary = summary.groupby(["bucket","policy"])["avg_power_w"].mean().unstack("policy")
    summary = summary.reindex(buckets)[main_policies]
    summary.rename(columns=policy_labels).plot(kind="bar", ax=ax,
        color=[policy_colors[p] for p in main_policies], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Average Power (W)", fontsize=11)
    ax.set_title(f"Average GPU Power — {model_name}", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_v2_power.png"), dpi=150)
    plt.close()

    # 4. Alpha sweep
    sweep_pols = [p for p in results_df["policy"].unique() if p.startswith("adaptive_")]
    if len(sweep_pols) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        bucket_colors = {"short": "#42A5F5", "medium": "#FFA726", "long": "#66BB6A"}

        sweep_data = []
        for pol in sorted(sweep_pols):
            alpha_val = float(pol.split("_")[1])
            for bucket in buckets:
                bsub = results_df[(results_df["policy"]==pol) & (results_df["bucket"]==bucket)]
                default_e = results_df[(results_df["policy"]=="default") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
                if len(bsub) > 0:
                    sweep_data.append({
                        "alpha": alpha_val, "bucket": bucket,
                        "saving_pct": (1 - bsub["energy_total_j"].mean()/default_e)*100,
                        "tbt_p95_ms": bsub["tbt_p95_ms"].mean(),
                        "delay_mean_ms": bsub["delay_mean_ms"].mean(),
                    })
        sdf = pd.DataFrame(sweep_data)

        # (a) Energy saving vs alpha
        ax = axes[0]
        for bucket in buckets:
            bdf = sdf[sdf["bucket"]==bucket]
            ax.plot(bdf["alpha"], bdf["saving_pct"], "o-", color=bucket_colors[bucket],
                    label=bucket.capitalize(), linewidth=2, markersize=8, markeredgecolor="black")
        # Static reference lines
        for bucket in buckets:
            static_e = results_df[(results_df["policy"]=="static") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
            default_e = results_df[(results_df["policy"]=="default") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
            ax.axhline(y=(1-static_e/default_e)*100, color=bucket_colors[bucket], linestyle="--", alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel("Energy Saving (%)", fontsize=11)
        ax.set_title("(a) Energy Saving vs Alpha\n(dashed = static 5ms)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (b) TBT P95 vs alpha
        ax = axes[1]
        for bucket in buckets:
            bdf = sdf[sdf["bucket"]==bucket]
            ax.plot(bdf["alpha"], bdf["tbt_p95_ms"], "o-", color=bucket_colors[bucket],
                    label=bucket.capitalize(), linewidth=2, markersize=8, markeredgecolor="black")
        ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="100ms SLO")
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel("TBT P95 (ms)", fontsize=11)
        ax.set_title("(b) Decode Latency vs Alpha", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (c) Mean delay vs alpha
        ax = axes[2]
        for bucket in buckets:
            bdf = sdf[sdf["bucket"]==bucket]
            ax.plot(bdf["alpha"], bdf["delay_mean_ms"], "o-", color=bucket_colors[bucket],
                    label=bucket.capitalize(), linewidth=2, markersize=8, markeredgecolor="black")
        ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="Static (5ms)")
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel("Mean Delay (ms)", fontsize=11)
        ax.set_title("(c) Mean Delay vs Alpha\n(dashed = static 5ms)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Calibrated Alpha Sweep — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "adaptive_v2_alpha_sweep.png"), dpi=150)
        plt.close()

    # 5. Per-token delay distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    alpha_colors = {"adaptive_0.05": "#C8E6C9", "adaptive_0.10": "#81C784",
                    "adaptive_0.12": "#4CAF50", "adaptive_0.15": "#2E7D32", "adaptive_0.20": "#1B5E20"}
    for idx, bucket in enumerate(buckets):
        ax = axes[idx]
        for pol in sorted(sweep_pols):
            subset = results_df[(results_df["policy"]==pol) & (results_df["bucket"]==bucket)]
            all_delays = []
            for _, row in subset.iterrows():
                all_delays.extend(row["delay_trace_ms"])
            if all_delays:
                label = f"α={pol.split('_')[1]}"
                ax.hist(all_delays, bins=25, alpha=0.5, label=label,
                        color=alpha_colors.get(pol, "#888"), edgecolor="black", linewidth=0.3)
        ax.axvline(x=5.0, color="red", linestyle="--", alpha=0.7, label="Static 5ms")
        ax.set_xlabel("Delay (ms)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{bucket.capitalize()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
    plt.suptitle(f"Per-Token Delay Distribution — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_v2_delay_dist.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {FIGURES_DIR}")


def main():
    print("=" * 60)
    print("Adaptive DVFS v2 (Calibrated) — Qwen2.5-7B")
    print(f"T_budget={T_BUDGET_MS}ms, d_max={CONFIG['d_max_ms']}ms")
    print(f"Alpha sweep: {CONFIG['alpha_sweep']}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No GPU!")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        print(f"NVML: {power_mw / 1000.0:.1f} W")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"NVML FAILED: {e}")

    model_name = CONFIG["model_name"]
    tokenizer, model = load_model_and_tokenizer(CONFIG["model_id"], CONFIG["use_4bit"])
    prompts = prepare_dataset(tokenizer, CONFIG)
    if not prompts:
        return
    warmup_gpu(model, tokenizer)

    all_results = []

    # ── Phase 1: Default + Static + Adaptive(α=0.12) ──
    policies = {
        "default":       {"policy": "default"},
        "static":        {"policy": "static", "static_delay_ms": CONFIG["static_decode_delay_ms"]},
        "adaptive_0.12": {"policy": "adaptive", "alpha": 0.12},
    }

    total_runs = len(policies) * sum(len(v) for v in prompts.values())
    run_idx = 0

    for policy_name, params in policies.items():
        print(f"\n{'#'*60}")
        print(f"# POLICY: {policy_name}")
        print(f"{'#'*60}")

        tasks = [(b, p) for b, pl in prompts.items() for p in pl]
        random.seed(42)
        random.shuffle(tasks)

        for bucket, prompt in tasks:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {policy_name:16s} {bucket:6s} | ", end="", flush=True)
            result = run_benchmark(
                model, tokenizer, prompt, bucket,
                policy=params["policy"], max_new_tokens=CONFIG["max_new_tokens"],
                static_delay_ms=params.get("static_delay_ms", 0.0),
                alpha=params.get("alpha", 0.12),
                T_budget_ms=T_BUDGET_MS, d_max_ms=CONFIG["d_max_ms"],
            )
            result["policy"] = policy_name
            print(f"E={result['energy_total_j']:.1f}J P={result['avg_power_w']:.1f}W "
                  f"d={result['delay_mean_ms']:.1f}ms TPS={result['tokens_per_sec']:.1f}")
            all_results.append(result)
            time.sleep(0.3)

    # ── Phase 2: Alpha sweep ──
    print(f"\n{'#'*60}")
    print(f"# ALPHA SWEEP")
    print(f"{'#'*60}")

    for alpha in CONFIG["alpha_sweep"]:
        pname = f"adaptive_{alpha}"
        if pname in policies:
            continue
        print(f"\n  α={alpha}")
        tasks = [(b, p) for b, pl in prompts.items() for p in pl]
        random.seed(42)
        random.shuffle(tasks)
        for bucket, prompt in tasks:
            print(f"    {pname:16s} {bucket:6s} | ", end="", flush=True)
            result = run_benchmark(
                model, tokenizer, prompt, bucket,
                policy="adaptive", max_new_tokens=CONFIG["max_new_tokens"],
                alpha=alpha, T_budget_ms=T_BUDGET_MS, d_max_ms=CONFIG["d_max_ms"],
            )
            result["policy"] = pname
            print(f"E={result['energy_total_j']:.1f}J d={result['delay_mean_ms']:.1f}ms")
            all_results.append(result)
            time.sleep(0.2)

    results_df = pd.DataFrame(all_results)

    # Save
    save_df = results_df.drop(columns=["tbt_trace_ms", "delay_trace_ms"], errors="ignore")
    save_df.to_csv(os.path.join(LOGS_DIR, f"adaptive_v2_{model_name}.csv"), index=False)

    # Summaries
    summary_cols = ["input_tokens", "generated_tokens", "ttft_ms", "tbt_p95_ms", "tbt_max_ms",
                    "tokens_per_sec", "energy_total_j", "avg_power_w", "edp",
                    "delay_mean_ms", "slo_violations"]

    for pol in ["default", "static", "adaptive_0.12"]:
        pdf = results_df[results_df["policy"] == pol]
        if len(pdf) == 0:
            continue
        print(f"\n{'='*90}")
        print(f"SUMMARY — {pol}")
        print(f"{'='*90}")
        print(pdf.groupby("bucket")[summary_cols].mean().round(3).to_string())

    # Energy comparison
    print(f"\n{'='*90}")
    print(f"ENERGY SAVINGS: Static vs Adaptive")
    print(f"{'='*90}")
    for bucket in ["short", "medium", "long"]:
        de = results_df[(results_df["policy"]=="default") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
        se = results_df[(results_df["policy"]=="static") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
        ae = results_df[(results_df["policy"]=="adaptive_0.12") & (results_df["bucket"]==bucket)]["energy_total_j"].mean()
        sp = results_df[(results_df["policy"]=="static") & (results_df["bucket"]==bucket)]["avg_power_w"].mean()
        ap = results_df[(results_df["policy"]=="adaptive_0.12") & (results_df["bucket"]==bucket)]["avg_power_w"].mean()
        ad = results_df[(results_df["policy"]=="adaptive_0.12") & (results_df["bucket"]==bucket)]["delay_mean_ms"].mean()
        dp = results_df[(results_df["policy"]=="default") & (results_df["bucket"]==bucket)]["avg_power_w"].mean()
        print(f"\n  {bucket:8s}:")
        print(f"    Default:        Energy={de:.1f}J  Power={dp:.1f}W")
        print(f"    Static (5ms):   Energy={se:.1f}J ({(1-se/de)*100:+.1f}%)  Power={sp:.1f}W")
        print(f"    Adaptive α=0.12: Energy={ae:.1f}J ({(1-ae/de)*100:+.1f}%)  Power={ap:.1f}W  AvgDelay={ad:.1f}ms")
        print(f"    → Adaptive {'WINS' if (1-ae/de) > (1-se/de) else 'LOSES'} by {abs((1-ae/de)*100 - (1-se/de)*100):.1f}pp")

    # Alpha sweep
    print(f"\n{'='*90}")
    print(f"ALPHA SWEEP SUMMARY")
    print(f"{'='*90}")
    default_e_all = results_df[results_df["policy"]=="default"]["energy_total_j"].mean()
    for alpha in CONFIG["alpha_sweep"]:
        pol = f"adaptive_{alpha}"
        pdf = results_df[results_df["policy"] == pol]
        if len(pdf) == 0:
            continue
        print(f"  α={alpha:>4}: Energy={pdf['energy_total_j'].mean():.1f}J "
              f"({(1-pdf['energy_total_j'].mean()/default_e_all)*100:+.1f}%)  "
              f"Power={pdf['avg_power_w'].mean():.1f}W  "
              f"AvgDelay={pdf['delay_mean_ms'].mean():.1f}ms  "
              f"TBT_P95={pdf['tbt_p95_ms'].mean():.1f}ms  "
              f"SLO_viol={int(pdf['slo_violations'].sum())}")

    # Save tables
    for pol in results_df["policy"].unique():
        pdf = results_df[results_df["policy"] == pol]
        pdf.groupby("bucket")[summary_cols].mean().round(3).to_csv(
            os.path.join(TABLES_DIR, f"summary_{model_name}_{pol}.csv"))

    generate_plots(results_df, model_name)
    print("\nAdaptive DVFS v2 complete!")


if __name__ == "__main__":
    main()
