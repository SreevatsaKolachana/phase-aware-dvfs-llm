"""
DVFS Simulation for Phase-Aware LLM Inference — Llama-2-7B (Full MHA)
=====================================================================
Llama-2-7B uses FULL Multi-Head Attention (32 query heads, 32 KV heads, GQA ratio = 1),
giving it the LARGEST KV cache among our 3 models:

  - Llama-2-7B:  n_kv=32 (GQA-1x, full MHA)  ← THIS MODEL
  - Mistral-7B:  n_kv=8  (GQA-4x)
  - Qwen2.5-7B:  n_kv=4  (GQA-7x)

All three are 7B-scale models, making this an apples-to-apples comparison
of how attention architecture affects phase-aware DVFS effectiveness.
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

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results_llama2")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIG = {
    "model_id": os.path.join(PROJECT_DIR, "models/Llama-2-7b-chat-hf"),
    "model_name": "Llama-2-7B",
    "max_new_tokens": 128,
    "use_4bit": True,
    "samples_per_bucket": 20,
    "hf_token": None,
    "buckets": {
        "short": (1, 64),
        "medium": (65, 256),
        "long": (257, 1024),
    },
    "throttle_delays_ms": {
        "light":  5.0,
        "medium": 10.0,
        "heavy":  20.0,
    },
    "default_throttle": "light",
}


# GPU POWER SAMPLER
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


# MODEL LOADING
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


# WARMUP
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


# DATASET
def prepare_dataset(tokenizer, config):
    prompts_file = os.path.join(DATA_DIR, f"sharegpt_bucketed_n{config['samples_per_bucket']}_dvfs_llama2.json")

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


# CORE: DVFS-AWARE BENCHMARK
def run_dvfs_benchmark(model, tokenizer, prompt, bucket, policy,
                       max_new_tokens=128, gpu_index=0,
                       prefill_delay_ms=0.0, decode_delay_ms=0.0):
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

    # PREFILL
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            use_cache=True,
        )

    torch.cuda.synchronize()

    if prefill_delay_ms > 0:
        time.sleep(prefill_delay_ms / 1000.0)

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

        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
            )

        torch.cuda.synchronize()

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

    # METRICS
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


# PLOTTING
def generate_dvfs_plots(results_df, model_name):
    policies = results_df["policy"].unique()
    buckets = ["short", "medium", "long"]
    policy_colors = {"default": "#2196F3", "static": "#FF5722", "phase_aware": "#4CAF50"}
    policy_labels = {"default": "Default", "static": "Static Throttle", "phase_aware": "Phase-Aware"}

    # 1. DVFS Policy Comparison
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

    # 2. Energy breakdown
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

    # 3. Avg power
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

    # 4. EDP
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

    # 5. Throttle sweep
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

            ax = axes[0, 0]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["energy_total_j_mean"],
                       yerr=sweep_summary["energy_total_j_std"], fmt="o-", color="#4CAF50",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
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

            ax = axes[0, 1]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["tbt_p95_ms_mean"],
                       yerr=sweep_summary["tbt_p95_ms_std"], fmt="o-", color="#E91E63",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7, label="100ms SLO")
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("TBT P95 (ms)", fontsize=11)
            ax.set_title("Decode Latency vs Throttle", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.errorbar(sweep_summary["decode_delay_ms"], sweep_summary["avg_power_w_mean"],
                       yerr=sweep_summary["avg_power_w_std"], fmt="s-", color="#FF9800",
                       linewidth=2, capsize=5, markersize=8, markeredgecolor="black")
            ax.set_xlabel("Decode Delay (ms)", fontsize=11)
            ax.set_ylabel("Average Power (W)", fontsize=11)
            ax.set_title("GPU Power vs Throttle", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

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


# MAIN
def main():
    print("=" * 60)
    print("DVFS Simulation Benchmark — Llama-2-7B (Full MHA)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    tokenizer, model = load_model_and_tokenizer(CONFIG["model_id"], CONFIG["use_4bit"], hf_token)

    # Print architecture info
    print(f"\nModel Architecture:")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")
    n_kv = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    print(f"  Num KV heads: {n_kv}")
    print(f"  GQA ratio: {model.config.num_attention_heads // n_kv}x")
    print(f"  Intermediate size: {model.config.intermediate_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")

    prompts = prepare_dataset(tokenizer, CONFIG)
    warmup_gpu(model, tokenizer)

    throttle_key = CONFIG["default_throttle"]
    delay_ms = CONFIG["throttle_delays_ms"][throttle_key]
    print(f"\nThrottle level: {throttle_key} ({delay_ms}ms decode delay)")

    policies = {
        "default":     {"prefill_delay_ms": 0.0,      "decode_delay_ms": 0.0},
        "static":      {"prefill_delay_ms": delay_ms,  "decode_delay_ms": delay_ms},
        "phase_aware": {"prefill_delay_ms": 0.0,       "decode_delay_ms": delay_ms},
    }

    all_results = []
    total_runs = len(policies) * sum(len(v) for v in prompts.values())
    run_idx = 0

    for policy_name, delays in policies.items():
        print(f"\n{'#'*60}")
        print(f"# POLICY: {policy_name}")
        print(f"# Prefill delay: {delays['prefill_delay_ms']}ms")
        print(f"# Decode delay:  {delays['decode_delay_ms']}ms")
        print(f"{'#'*60}")

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

    # Throttle sweep
    print(f"\n{'#'*60}")
    print(f"# THROTTLE SWEEP (phase_aware at all delay levels)")
    print(f"{'#'*60}")

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

    # Save
    save_df = results_df.drop(columns=["tbt_trace_ms"], errors="ignore")
    csv_path = os.path.join(LOGS_DIR, f"dvfs_results_{model_name}.csv")
    save_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    summary_cols = [
        "input_tokens", "generated_tokens", "ttft_ms", "tbt_p95_ms",
        "tokens_per_sec", "ms_per_token",
        "energy_prefill_j", "energy_decode_j", "energy_total_j",
        "energy_per_token_mj", "avg_power_w", "edp"
    ]

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

    # Energy savings
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

    generate_dvfs_plots(results_df, model_name)
    print("\nDVFS simulation complete!")


if __name__ == "__main__":
    main()
