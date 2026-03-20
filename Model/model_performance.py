import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", device_map="cuda")

# ── Hardware specs ─────────────────────────────────────────────────────────────
# BW1 = PCIe bandwidth (lower, e.g. CPU→GPU or NVLink bottleneck in multi-GPU)
# BW2 = HBM peak bandwidth (higher, on-chip VRAM bandwidth)
# Change these to match your GPU (defaults: A100 SXM)
PEAK_FLOPS = 312e12   # 312 TFLOP/s bf16  (A100 SXM)
BW2        = 2.0e12   # 2.0 TB/s HBM2e    (A100 SXM) — high bandwidth
BW1        = 0.5e12   # 0.5 TB/s          — lower bandwidth scenario (e.g. PCIe, or throttled)

ridge_bw2 = PEAK_FLOPS / BW2   # compute/memory crossover for BW2
ridge_bw1 = PEAK_FLOPS / BW1   # compute/memory crossover for BW1

# ── Measure model sizes ────────────────────────────────────────────────────────
num_params   = sum(p.numel() for p in model.parameters())
weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

# ── Build inputs ───────────────────────────────────────────────────────────────
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
    {"type": "text", "text": "What animal is on the candy?"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

# ── Prefill: run forward pass, time it ────────────────────────────────────────
WARMUP, RUNS = 2, 5
for _ in range(WARMUP):
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
torch.cuda.synchronize()
prefill_time = (time.perf_counter() - t0) / RUNS

prefill_seq   = inputs["input_ids"].shape[1]
prefill_flops = 2 * num_params * prefill_seq
prefill_bytes = weight_bytes  # weights dominate; KV write is small relative to seq_len×params
prefill_ai    = prefill_flops / prefill_bytes
prefill_perf  = prefill_flops / prefill_time   # achieved FLOP/s

# ── Decode: single token step ──────────────────────────────────────────────────
past_kv      = prefill_out.past_key_values
last_token   = inputs["input_ids"][:, -1:]
past_seq_len = inputs["input_ids"].shape[1]
attn_mask    = torch.ones(1, past_seq_len + 1, dtype=torch.long, device=model.device)
position_ids = torch.tensor([[past_seq_len]], dtype=torch.long, device=model.device)

kv_cache_bytes = sum(
    t.numel() * t.element_size()
    for t in past_kv.key_cache + past_kv.value_cache
    if t is not None
)
decode_bytes = weight_bytes + kv_cache_bytes

for _ in range(WARMUP):
    with torch.no_grad():
        model(input_ids=last_token, attention_mask=attn_mask,
              position_ids=position_ids, past_key_values=past_kv, use_cache=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        model(input_ids=last_token, attention_mask=attn_mask,
              position_ids=position_ids, past_key_values=past_kv, use_cache=True)
torch.cuda.synchronize()
decode_time = (time.perf_counter() - t0) / RUNS

decode_flops = 2 * num_params * 1
decode_ai    = decode_flops / decode_bytes
decode_perf  = decode_flops / decode_time   # achieved FLOP/s

print(f"Prefill  AI={prefill_ai:.0f} FLOP/B  perf={prefill_perf/1e12:.2f} TFLOP/s  time={prefill_time*1000:.0f}ms")
print(f"Decode   AI={decode_ai:.2f} FLOP/B  perf={decode_perf/1e12:.4f} TFLOP/s  time={decode_time*1000:.2f}ms")
print(f"Ridge BW2={ridge_bw2:.0f}  Ridge BW1={ridge_bw1:.0f}  FLOP/B")

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xscale("log")
ax.set_yscale("log")

ai_vals = np.logspace(-2, 4, 500)

# Roofline for BW1 and BW2
roof_bw1 = np.minimum(BW1 * ai_vals, PEAK_FLOPS)
roof_bw2 = np.minimum(BW2 * ai_vals, PEAK_FLOPS)

# ── Colored regions ────────────────────────────────────────────────────────────
# Red:    AI < ridge_bw1  → memory bound at BOTH bandwidths
# Yellow: ridge_bw1 ≤ AI < ridge_bw2  → compute bound at BW2, memory bound at BW1
# Green:  AI ≥ ridge_bw2  → compute bound at BOTH bandwidths
ax.axvspan(ai_vals[0],  ridge_bw1,  alpha=0.18, color="red",    zorder=0, label="BW bound at BW1 & BW2")
ax.axvspan(ridge_bw1,   ridge_bw2,  alpha=0.18, color="gold",   zorder=0, label="BW bound at BW1, Compute bound at BW2")
ax.axvspan(ridge_bw2,   ai_vals[-1],alpha=0.18, color="green",  zorder=0, label="Compute bound at BW1 & BW2")

# Roofline curves
ax.plot(ai_vals, roof_bw1, color="cornflowerblue", linewidth=2, linestyle="--", label=f"BW₁ = {BW1/1e12:.1f} TB/s roofline")
ax.plot(ai_vals, roof_bw2, color="magenta",        linewidth=2.5,              label=f"BW₂ = {BW2/1e12:.1f} TB/s roofline")

# Peak compute line
ax.axhline(PEAK_FLOPS, color="gray", linestyle=":", linewidth=1)
ax.text(1.5e3, PEAK_FLOPS * 1.12, f"Peak {PEAK_FLOPS/1e12:.0f} TFLOP/s", fontsize=9, color="gray")

# Ridge point markers on x-axis
ax.axvline(ridge_bw1, color="cornflowerblue", linestyle=":", linewidth=1)
ax.axvline(ridge_bw2, color="magenta",        linestyle=":", linewidth=1)
ax.text(ridge_bw1 * 1.05, 5e9,
        f"Ridge BW₁\n{ridge_bw1:.0f} FLOP/B", fontsize=8, color="cornflowerblue",
        rotation=90, va="bottom")
ax.text(ridge_bw2 * 1.05, 5e9,
        f"Ridge BW₂\n{ridge_bw2:.0f} FLOP/B", fontsize=8, color="magenta",
        rotation=90, va="bottom")

# ── Plot actual measured points ────────────────────────────────────────────────
# Achieved performance (solid dot) vs theoretical roofline ceiling (open circle)
prefill_roof = min(BW2 * prefill_ai, PEAK_FLOPS)
decode_roof  = min(BW2 * decode_ai,  PEAK_FLOPS)

# Open circles = theoretical max on BW2 roofline
ax.scatter(prefill_ai, prefill_roof, s=130, facecolors="none", edgecolors="royalblue", linewidths=2, zorder=6)
ax.scatter(decode_ai,  decode_roof,  s=130, facecolors="none", edgecolors="tomato",    linewidths=2, zorder=6)

# Solid dots = actual measured performance
ax.scatter(prefill_ai, prefill_perf, s=130, color="royalblue", zorder=7)
ax.scatter(decode_ai,  decode_perf,  s=130, color="tomato",    zorder=7)

# Vertical dashed arrows from achieved → roofline ceiling
ax.annotate("", xy=(prefill_ai, prefill_roof), xytext=(prefill_ai, prefill_perf),
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.5, linestyle="dashed"))
ax.annotate("", xy=(decode_ai, decode_roof), xytext=(decode_ai, decode_perf),
            arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5, linestyle="dashed"))

# Labels
ax.annotate(f"Prefill\n{prefill_ai:.0f} FLOP/B\n{prefill_perf/1e12:.1f} TFLOP/s (achieved)\n{prefill_roof/1e12:.0f} TFLOP/s (roof)",
            xy=(prefill_ai, prefill_perf),
            xytext=(prefill_ai * 0.08, prefill_perf * 8),
            arrowprops=dict(arrowstyle="->", color="royalblue"),
            color="royalblue", fontsize=8.5,
            annotation_clip=False)
ax.annotate(f"Decode\n{decode_ai:.1f} FLOP/B\n{decode_perf/1e12:.4f} TFLOP/s (achieved)\n{decode_roof/1e12:.3f} TFLOP/s (roof)",
            xy=(decode_ai, decode_perf),
            xytext=(decode_ai * 3, decode_perf * 0.08),
            arrowprops=dict(arrowstyle="->", color="tomato"),
            color="tomato", fontsize=8.5)

ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
ax.set_ylabel("Performance (FLOP/s)  [realized, log-scale]", fontsize=12)
ax.set_title("Roofline Model — Qwen3.5-2B  (Prefill vs Decode)\n"
             r"Performance = min(Peak FLOPs/s,  AI × BW)", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.set_xlim(1e-2, 1e4)
ax.set_ylim(5e9, PEAK_FLOPS * 5)  # fixed y-range so both points are always visible

plt.tight_layout()
plt.savefig("/home/roofline_bw.png", dpi=150)
print("Saved → /home/roofline_bw.png")