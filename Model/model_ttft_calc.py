# Load model directly
import time
import threading
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", device_map="cuda")

# ── Memory: Weights ──────────────────────────────────────────────────────────
# Each parameter is stored as float32 (4 bytes) or float16/bf16 (2 bytes).
# Formula: num_params × bytes_per_param
weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
# Buffers (e.g. LayerNorm running stats) are non-trainable tensors also in VRAM
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
weights_mb = (weight_bytes + buffer_bytes) / 1024**2

# ── Memory: Gradients ────────────────────────────────────────────────────────
# During training, every trainable param gets a grad tensor of the same dtype/size.
# Formula: num_trainable_params × bytes_per_param  (0 at inference, shown for reference)
grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
grads_mb = grad_bytes / 1024**2

# ── Memory: Optimizer states ─────────────────────────────────────────────────
# Adam/AdamW keeps 2 moment vectors (fp32) per trainable param → 8 bytes/param.
# SGD keeps 1 momentum buffer → 4 bytes/param.
# Formula (AdamW): num_trainable_params × 8
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
optimizer_mb = (trainable_params * 8) / 1024**2  # assumes AdamW, fp32 states

# ── Memory: Activations ──────────────────────────────────────────────────────
# Activations depend on batch_size × seq_len × hidden_dim × num_layers × dtype.
# They are created at runtime; we measure actual VRAM before and after the forward pass.
torch.cuda.synchronize()
vram_before_mb = torch.cuda.memory_allocated() / 1024**2

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

torch.cuda.synchronize()
vram_after_inputs_mb = torch.cuda.memory_allocated() / 1024**2
activations_mb = vram_after_inputs_mb - vram_before_mb  # input tensors + KV-cache allocation

streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # yields decoded text tokens one by one as they are generated
gen_kwargs = {**inputs, "max_new_tokens": 40, "streamer": streamer}  # pack all generate() arguments into a dict for passing to the thread

thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)  # create a background thread so generate() doesn't block the main thread
start = time.perf_counter()  # record the start time just before launching generation (used for TTFT)
thread.start()  # start generation in the background; tokens will be pushed into the streamer as they are produced

timestamps = []  # will hold the wall-clock time at which each token was received
tokens = []      # will hold the decoded text of each token
for token in streamer:          # iterate; blocks until the next token is ready, then unblocks immediately
    timestamps.append(time.perf_counter())  # capture the exact time this token arrived
    tokens.append(token)        # collect the token text for final output

thread.join()  # wait for the generation thread to fully finish before proceeding

torch.cuda.synchronize()
vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2  # peak VRAM used across the full run

# ── Latency metrics ──────────────────────────────────────────────────────────
num_tokens     = len(timestamps)
total_time     = timestamps[-1] - start
ttft           = timestamps[0] - start
itl            = (timestamps[-1] - timestamps[0]) / (num_tokens - 1) if num_tokens > 1 else 0

# ── Throughput ───────────────────────────────────────────────────────────────
# Tokens generated per second during the decode phase (excludes prefill/TTFT).
# Formula: (num_output_tokens - 1) / (last_token_time - first_token_time)
decode_throughput = (num_tokens - 1) / (timestamps[-1] - timestamps[0]) if num_tokens > 1 else 0
# End-to-end throughput includes prefill time as well.
e2e_throughput = num_tokens / total_time

print("".join(tokens))

print("\n─── Latency ───────────────────────────────")
print(f"  TTFT              : {ttft * 1000:.1f} ms")
print(f"  ITL               : {itl  * 1000:.1f} ms/token")
print(f"  Tokens generated  : {num_tokens}")

print("\n─── Throughput ────────────────────────────")
print(f"  Decode throughput : {decode_throughput:.1f} tokens/s  (decode phase only)")
print(f"  E2E throughput    : {e2e_throughput:.1f} tokens/s  (including prefill)")

print("\n─── Memory ────────────────────────────────")
print(f"  Weights + Buffers : {weights_mb:.1f} MB   (model params loaded in VRAM)")
print(f"  Gradients         : {grads_mb:.1f} MB   (same size as weights; 0 during inference)")
print(f"  Optimizer states  : {optimizer_mb:.1f} MB   (AdamW: 2× fp32 moments per trainable param)")
print(f"  Activations/KV    : {activations_mb:.1f} MB   (input tensors + KV-cache, measured at runtime)")
print(f"  Peak VRAM (total) : {vram_peak_mb:.1f} MB")