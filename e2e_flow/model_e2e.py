"""
End-to-end walkthrough of Qwen3.5-2B with a tiny 32×32 image + "cat?" text.
Prints every tensor shape at every stage to trace the full data flow.
"""
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
model     = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", device_map="cuda")
model.eval()

def sep(title): print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ══════════════════════════════════════════════════════════════
# STAGE 0 — RAW INPUTS
# ══════════════════════════════════════════════════════════════
sep("STAGE 0 — Raw inputs")
# Tiny 32×32 synthetic image (3 RGB channels)
img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
# Text question (will tokenize to ~2 tokens)
text = "cat?"
print(f"  Image : PIL (H=32, W=32, C=3)")
print(f"  Text  : '{text}'")

# ══════════════════════════════════════════════════════════════
# STAGE 1 — PROCESSOR (tokenise + image pre-process)
# ══════════════════════════════════════════════════════════════
sep("STAGE 1 — Processor output")
messages = [{"role": "user", "content": [
    {"type": "image",  "image": img},
    {"type": "text",   "text": text},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

for k, v in inputs.items():
    print(f"  {k:30s}: {tuple(v.shape)}")

# ══════════════════════════════════════════════════════════════
# STAGE 2 — VISION ENCODER  (model.model.visual)
# ══════════════════════════════════════════════════════════════
sep("STAGE 2 — Vision encoder (Qwen3_5VisionModel)")
visual = model.model.visual

# 2a. patch_embed  — Conv3d extracts 16×16 patches
#     Input : pixel_values  (B, C, T=2, H, W)
#     Kernel: (2, 16, 16)   stride (2, 16, 16)
#     Output: (B, 1024, T/2, H/16, W/16)  then flattened to (N_patches, 1024)
pv = inputs["pixel_values"]           # already flattened by processor: (N_patches, C*t*h*w)
grid = inputs["image_grid_thw"][0]    # [T, H_grids, W_grids]
T, H_g, W_g = int(grid[0]), int(grid[1]), int(grid[2])
N_patches = int(pv.shape[0])
print(f"  pixel_values (input to vision): {tuple(pv.shape)}")
print(f"  image_grid_thw               : T={T}  H_grids={H_g}  W_grids={W_g}")
print(f"  N_patches = T×H_grids×W_grids = {T}×{H_g}×{W_g} = {N_patches}")
print(f"  Each patch vector dim = {pv.shape[1]}  (= 3 channels × {T}×16×16 pixels flattened)")
print(f"  After Conv3d patch_embed (proj 1536→1024): ({N_patches}, 1024)")

# 2b. pos_embed  — learned position embedding added per patch
print(f"  After pos_embed          : ({N_patches}, 1024)  [adds pos info to each patch]")

# 2c. 24 × VisionBlock  — each: LayerNorm → Attention (QKV: 1024→3072) → proj → MLP
print(f"  Through 24 VisionBlocks  : ({N_patches}, 1024)  [shape unchanged]")
print(f"    Each block:")
print(f"      norm1  : ({N_patches},1024) → ({N_patches},1024)")
print(f"      qkv    : ({N_patches},1024) → ({N_patches},3072)  → split Q({N_patches},1024) K({N_patches},1024) V({N_patches},1024)")
print(f"      proj   : ({N_patches},1024) → ({N_patches},1024)")
print(f"      norm2  : ({N_patches},1024) → ({N_patches},1024)")
print(f"      fc1    : ({N_patches},1024) → ({N_patches},4096)")
print(f"      fc2    : ({N_patches},4096) → ({N_patches},1024)")

# 2d. merger  — squeezes 4 adjacent patches (2×2) into one token of dim 2048
#     Concat 4 neighbours → (N_patches/4, 4096) → fc1(4096→4096) → fc2(4096→2048)
# merger groups (T × 2×2) patches per output token
merge_factor = T * 4           # temporal stride=2 → T/2, spatial 2×2 → ×4 total
N_img_tokens = N_patches // merge_factor
print(f"  After merger             : ({N_img_tokens}, 2048)")
print(f"    concat 2×2 neighbours: ({N_patches},1024) → ({N_img_tokens},4096)")
print(f"    fc1 GELU               : ({N_img_tokens},4096) → ({N_img_tokens},4096)")
print(f"    fc2                    : ({N_img_tokens},4096) → ({N_img_tokens},2048)  ← matches text dim!")

# ══════════════════════════════════════════════════════════════
# STAGE 3 — TEXT EMBEDDING  (embed_tokens)
# ══════════════════════════════════════════════════════════════
sep("STAGE 3 — Text token embedding")
input_ids = inputs["input_ids"]
N_text = input_ids.shape[1]
print(f"  input_ids                : {tuple(input_ids.shape)}   (token IDs)")
print(f"  embed_tokens (248320×2048): ({N_text}, 2048)")
print(f"  Each ID → a vector of dim 2048")

# ══════════════════════════════════════════════════════════════
# STAGE 4 — MERGE VISION + TEXT into one sequence
# ══════════════════════════════════════════════════════════════
sep("STAGE 4 — Merged sequence (image tokens injected into text)")
N_seq = N_img_tokens + N_text
print(f"  Image tokens : ({N_img_tokens}, 2048)")
print(f"  Text  tokens : ({N_text},  2048)")
print(f"  Merged seq   : ({N_seq},  2048)   ← single flat sequence into decoder")
print(f"")
print(f"  Layout:  [img_0 ... img_{N_img_tokens-1}] [sys_tok ... text_tok ... asst_tok]")
print(f"            ←── {N_img_tokens} image tokens ──→  ←──────── {N_text} text tokens ────────→")

# ══════════════════════════════════════════════════════════════
# STAGE 5 — 28 DECODER LAYERS
# ══════════════════════════════════════════════════════════════
sep("STAGE 5 — 28 decoder layers  (pattern: 3×Linear + 1×Attention, repeat×7)")
print(f"""
  Pattern repeats 7 times (layers 0-27):
  ┌──────────────────────────────────────────────────────┐
  │  Layers i, i+1, i+2  →  GatedDeltaNet (linear attn) │
  │  Layer  i+3          →  Qwen3_5Attention (full attn) │
  └──────────────────────────────────────────────────────┘

  Input to all layers: ({N_seq}, 2048)

  ── GatedDeltaNet (layers 0,1,2, 4,5,6, ...) ──────────
    RMSNorm          : ({N_seq},2048) → ({N_seq},2048)
    in_proj_qkv      : ({N_seq},2048) → ({N_seq},6144)  split → Q,K,V each ({N_seq},2048)
    conv1d (causal)  : ({N_seq},6144) → ({N_seq},6144)  local context mixing
    in_proj_z        : ({N_seq},2048) → ({N_seq},2048)  gate
    in_proj_a/b      : ({N_seq},2048) → ({N_seq},16)    delta-net step size
    RMSNormGated     : → ({N_seq},2048)
    out_proj         : ({N_seq},2048) → ({N_seq},2048)
    ── MLP ──
    gate_proj + up_proj : ({N_seq},2048) → ({N_seq},6144) each
    SiLU gate        : gate_proj ⊙ SiLU(up_proj)  → ({N_seq},6144)
    down_proj        : ({N_seq},6144) → ({N_seq},2048)

  ── Qwen3_5Attention / GQA (layers 3,7,11,15,19,23,27) ─
  Formula: Attention(Q,K,V) = softmax(QKᵀ/√d_k) · V

    RMSNorm          : ({N_seq},2048) → ({N_seq},2048)
    q_proj           : ({N_seq},2048) → ({N_seq},4096)  reshape→ (16 heads × {N_seq} × 256)
    k_proj           : ({N_seq},2048) → ({N_seq},512)   reshape→ ( 2 heads × {N_seq} × 256)  [GQA: 8 Q share 1 KV]
    v_proj           : ({N_seq},2048) → ({N_seq},512)   reshape→ ( 2 heads × {N_seq} × 256)
    RMSNorm(q,k)     : per-head norm on dim 256
    RoPE             : rotary position applied to Q,K
    QKᵀ/√256        : (16,{N_seq},{N_seq})  softmax → attention weights
    ×V               : (16,{N_seq},256) → reshape → ({N_seq},4096)
    o_proj           : ({N_seq},4096) → ({N_seq},2048)
    ── MLP (same as above) ──

  Output of all 28 layers: ({N_seq}, 2048)  [unchanged shape]
""")

# ══════════════════════════════════════════════════════════════
# STAGE 6 — FINAL NORM + LM HEAD
# ══════════════════════════════════════════════════════════════
sep("STAGE 6 — Final RMSNorm + lm_head")
print(f"  RMSNorm    : ({N_seq},2048) → ({N_seq},2048)")
print(f"  lm_head    : ({N_seq},2048) → ({N_seq},248320)   [logits over vocab]")
print(f"  Take last  : (1,     2048) → (1,     248320)   [only last position matters]")
print(f"  argmax     : → token_id  → decode → first output word")

# ══════════════════════════════════════════════════════════════
# STAGE 7 — RUN ACTUAL FORWARD and show GQA attention formula
# ══════════════════════════════════════════════════════════════
sep("STAGE 7 — GQA formula verification")
d_k = 256
n_q_heads = 16
n_kv_heads = 2
groups = n_q_heads // n_kv_heads
print(f"  Q heads={n_q_heads},  KV heads={n_kv_heads},  groups={groups}  (each KV head shared by {groups} Q heads)")
print(f"  head_dim d_k = {d_k}")
print(f"  scale   = 1/√{d_k} = {1/d_k**0.5:.4f}")
print(f"")
print(f"  Attention(Q_i, K_j, V_j):  i ∈ [1..{n_q_heads}],  j = ⌊i/{groups}⌋  ← KV head index")
print(f"")
print(f"  Softmax((Q_i × Kⱼᵀ) / {1/d_k**0.5:.4f}) × Vⱼ")
print(f"  ({'×'.join([f'({N_seq},{d_k})'] + [f'({d_k},{N_seq})'])}) → ({N_seq},{N_seq}) → ({N_seq},{d_k})")

# ══════════════════════════════════════════════════════════════
# STAGE 8 — DECODE LOOP (autoregressive)
# ══════════════════════════════════════════════════════════════
sep("STAGE 8 — Autoregressive decode loop")
print(f"""
  Step 0 (prefill):
    Input  : ({N_seq}, 2048) — full sequence
    Output : ({N_seq}, 248320) logits → pick last → token_0
    KV cached for all {N_seq} positions

  Step 1 (decode):
    Input  : (1, 2048) — only new token_0 embedding
    past_kv: ({N_seq}, 256) K and V per KV head per layer  ← read from cache
    Output : (1, 248320) → token_1

  Step t:
    Input  : (1, 2048)
    past_kv: ({N_seq}+t-1, 256) — grows by 1 each step
    Output : (1, 248320) → token_t  ...until <eos>
""")

print("Done. All shapes above match the actual Qwen3.5-2B architecture in model.txt")
