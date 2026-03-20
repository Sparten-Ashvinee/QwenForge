import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", device_map="cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# ── GPU hardware roof numbers ────────────────────────────────────────────────
# Read peak values for your GPU from nvidia-smi / spec sheet and set below.
# Defaults: A100-80GB SXM  (change to your GPU if different)
PEAK_TFLOPS_BF16  = 312e12   # A100: 312 TFLOPS bf16  (use 77.97e12 for 3090, 165e12 for H100 PCIe)
PEAK_BW_BYTES_SEC = 2e12     # A100: 2 TB/s HBM2e      (use 936e9 for 3090, 3.35e12 for H100 SXM)
ridge_point = PEAK_TFLOPS_BF16 / PEAK_BW_BYTES_SEC  # FLOP/byte at which compute roof = memory roof

# ── Measure PREFILL FLOPs + bytes ────────────────────────────────────────────
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()

with torch.no_grad():
    prefill_out = model(**inputs, use_cache=True)

torch.cuda.synchronize()
mem_after = torch.cuda.memory_allocated()

# FlopCounterMode breaks on GQA (Q heads ≠ KV heads) in this model.
# Use the standard transformer approximation instead:
#   FLOPs ≈ 2 × num_params × seq_len
# This holds because every parameter participates in one matmul with 2 FLOPs (mul+add) per token.
# Vision encoder is included via model.parameters(); image patches ≈ prefix_len tokens.
num_params    = sum(p.numel() for p in model.parameters())
prefill_seq   = inputs["input_ids"].shape[1]
prefill_flops = 2 * num_params * prefill_seq             # total FLOPs for the forward pass
prefill_bytes = (mem_after - mem_before)                # bytes written to VRAM (KV cache + outputs)
# Also add weight reads: every parameter is read once per forward pass
weight_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
prefill_bytes_total = abs(prefill_bytes) + weight_bytes  # bytes moved = weights read + activations written
prefill_ai    = prefill_flops / prefill_bytes_total      # arithmetic intensity (FLOP/byte)

# ── Measure DECODE FLOPs + bytes (single step) ───────────────────────────────
# Pass the last generated token back in with the full KV cache (simulates one decode step).
past_kv      = prefill_out.past_key_values
last_token   = inputs["input_ids"][:, -1:]              # shape (1, 1) — single new token
past_seq_len = inputs["input_ids"].shape[1]             # number of tokens already in KV cache
# attention_mask must cover past tokens + 1 new token so position IDs are derived correctly
attn_mask    = torch.ones(1, past_seq_len + 1, dtype=torch.long, device=model.device)
# position_ids tells the model this new token sits at position past_seq_len (0-indexed)
position_ids = torch.tensor([[past_seq_len]], dtype=torch.long, device=model.device)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem_before_dec = torch.cuda.memory_allocated()

with torch.no_grad():
    dec_out = model(input_ids=last_token, attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=past_kv, use_cache=True)

torch.cuda.synchronize()
mem_after_dec = torch.cuda.memory_allocated()

# Decode: 1 token, same 2N rule → 2 × num_params × 1
decode_flops  = 2 * num_params * 1
# Bytes moved in decode = weights read (all layers) + KV cache read (all layers × seq_len)
kv_cache_bytes = sum(
    t.numel() * t.element_size()
    for t in past_kv.key_cache + past_kv.value_cache
    if t is not None   # linear-attention layers (GatedDeltaNet) have no KV cache → None
)
decode_bytes_total = weight_bytes + kv_cache_bytes
decode_ai = decode_flops / decode_bytes_total           # arithmetic intensity (FLOP/byte)

# ── Measure actual achieved FLOP/s ───────────────────────────────────────────
WARMUP = 2
RUNS   = 5

# Prefill achieved FLOP/s
for _ in range(WARMUP):
    with torch.no_grad():
        model(**inputs, use_cache=False)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        model(**inputs, use_cache=False)
torch.cuda.synchronize()
prefill_time   = (time.perf_counter() - t0) / RUNS
prefill_flops_sec = prefill_flops / prefill_time        # achieved FLOP/s

# Decode achieved FLOP/s
for _ in range(WARMUP):
    with torch.no_grad():
        model(input_ids=last_token, attention_mask=attn_mask,
              position_ids=position_ids,
              past_key_values=past_kv, use_cache=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        model(input_ids=last_token, attention_mask=attn_mask,
              position_ids=position_ids,
              past_key_values=past_kv, use_cache=True)
torch.cuda.synchronize()
decode_time   = (time.perf_counter() - t0) / RUNS
decode_flops_sec = decode_flops / decode_time           # achieved FLOP/s

# ── Print stats ───────────────────────────────────────────────────────────────
print(f"Ridge point          : {ridge_point:.1f}  FLOP/byte")
print(f"Prefill  AI          : {prefill_ai:.1f}  FLOP/byte  ({'compute' if prefill_ai > ridge_point else 'memory'} bound)")
print(f"Decode   AI          : {decode_ai:.2f}  FLOP/byte  ({'compute' if decode_ai > ridge_point else 'memory'} bound)")
print(f"Prefill  achieved    : {prefill_flops_sec/1e12:.1f}  TFLOP/s  ({prefill_flops/1e9:.0f} GFLOPs in {prefill_time*1000:.0f} ms)")
print(f"Decode   achieved    : {decode_flops_sec/1e12:.3f} TFLOP/s  ({decode_flops/1e6:.0f} MFLOPs in {decode_time*1000:.1f} ms)")

# ── Plot roofline ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Roofline boundary
ai_range = [1e-2, ridge_point, 1e4]
roof     = [min(PEAK_BW_BYTES_SEC * ai, PEAK_TFLOPS_BF16) for ai in ai_range]
ax.loglog(ai_range, roof, "k-", linewidth=2, label="Roofline")

# Annotate the two roofs
ax.axhline(PEAK_TFLOPS_BF16, color="gray", linestyle="--", linewidth=0.8)
ax.text(1e3, PEAK_TFLOPS_BF16 * 1.15, f"Compute roof  {PEAK_TFLOPS_BF16/1e12:.0f} TFLOPS",
        fontsize=9, color="gray")
ax.axvline(ridge_point, color="gray", linestyle=":", linewidth=0.8)
ax.text(ridge_point * 1.05, PEAK_TFLOPS_BF16 * 0.02,
        f"Ridge {ridge_point:.0f} FLOP/B", fontsize=9, color="gray", rotation=90, va="bottom")

# Prefill point
ax.scatter(prefill_ai, prefill_flops_sec, s=120, color="royalblue", zorder=5)
ax.annotate(f"Prefill\n{prefill_ai:.0f} FLOP/B\n{prefill_flops_sec/1e12:.1f} TFLOP/s",
            xy=(prefill_ai, prefill_flops_sec),
            xytext=(prefill_ai * 0.15, prefill_flops_sec * 2),
            arrowprops=dict(arrowstyle="->", color="royalblue"),
            color="royalblue", fontsize=9)

# Decode point
ax.scatter(decode_ai, decode_flops_sec, s=120, color="tomato", zorder=5)
ax.annotate(f"Decode\n{decode_ai:.1f} FLOP/B\n{decode_flops_sec/1e12:.3f} TFLOP/s",
            xy=(decode_ai, decode_flops_sec),
            xytext=(decode_ai * 0.15, decode_flops_sec * 0.1),
            arrowprops=dict(arrowstyle="->", color="tomato"),
            color="tomato", fontsize=9)

ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
ax.set_ylabel("Performance (FLOP/s)", fontsize=11)
ax.set_title("Roofline Model — Qwen3.5-2B  (Prefill vs Decode)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("/home/roofline.png", dpi=150)
print("\nSaved → /home/roofline.png")