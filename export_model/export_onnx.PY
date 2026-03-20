"""
Export Qwen3.5-2B submodules to ONNX for Netron visualization.
The full model can't be traced end-to-end (hybrid cache uses dynamic Python control flow),
so we export three self-contained pieces that cover the full architecture:
  1. vision_encoder.onnx  — patch_embed + 24 VisionBlocks + merger
  2. attn_layer.onnx      — one Qwen3_5Attention decoder layer (full GQA attention + MLP)
  3. linear_attn_layer.onnx — one Qwen3_5GatedDeltaNet decoder layer (linear attention + MLP)
Each file can be opened independently in https://netron.app/
"""
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from pathlib import Path

OUT_DIR = Path("/home/onnx_exports")
OUT_DIR.mkdir(exist_ok=True)

print("Loading model (cpu, fp32, eager attention for ONNX export)...")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-2B",
    device_map="cpu",
    dtype=torch.float32,        # ONNX export requires fp32
    attn_implementation="eager", # avoids F.scaled_dot_product_attention with enable_gqa=True
                                 # which the legacy jit.trace ONNX exporter can't convert
)
model.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT 1: Vision Encoder
# Input:  pixel_values (N_patches, 1536) — pre-flattened patches from processor
# Output: image_features (N_img_tokens, 2048) — merged tokens ready for LM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Exporting vision encoder...")

class VisionEncoderWrapper(nn.Module):
    """Wraps model.model.visual to accept simple (N, C) pixel patch tensor."""
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw):
        # visual forward expects pixel_values (N_patches, flat_dim) and grid_thw (1, 3)
        return self.visual(pixel_values, grid_thw)

vision_wrapper = VisionEncoderWrapper(model.model.visual)
vision_wrapper.eval()

# Dummy inputs: grid_thw=[T=2, H_grids=2, W_grids=2] → T*H*W = 8 patches total
# Each patch is 1536 dims (3 channels × 2 temporal × 16×16 spatial)
N_patches = 8  # must equal T * H_grids * W_grids = 2*2*2
pixel_values_dummy = torch.randn(N_patches, 1536)
grid_thw_dummy     = torch.tensor([[2, 2, 2]], dtype=torch.long)  # T=2, H_grids=2, W_grids=2

with torch.no_grad():
    torch.onnx.export(
        vision_wrapper,
        (pixel_values_dummy, grid_thw_dummy),
        str(OUT_DIR / "vision_encoder.onnx"),
        input_names=["pixel_values", "grid_thw"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "N_patches"},   # number of patches varies with image size
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,   # use legacy jit.trace exporter; new dynamo exporter fails on
                        # data-dependent torch.linspace in fast_pos_embed_interpolate
    )
print(f"  Saved → {OUT_DIR}/vision_encoder.onnx")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT 2: Full-attention decoder layer (Qwen3_5Attention + MLP)
# Layers 3, 7, 11, 15, 19, 23, 27 — GQA attention (16 Q heads, 2 KV heads)
# Input:  hidden_states (seq_len, hidden_dim=2048)
# Output: hidden_states (seq_len, 2048)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Exporting full-attention decoder layer (layer 3)...")

class AttnLayerWrapper(nn.Module):
    """Wraps one full-attention decoder layer. Computes RoPE embeddings internally."""
    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, attention_mask, position_ids):
        # position_ids shape: (1, seq_len) — Qwen3_5TextModel uses 4D mrope internally,
        # but for a single-layer export we pass 2D and let rotary_emb handle it.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        out = self.layer(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=None,
        )
        return out  # (batch, seq_len, hidden_dim)

attn_layer  = model.model.language_model.layers[3]  # layer 3 = first full-attention layer
rotary_emb  = model.model.language_model.rotary_emb
attn_wrapper = AttnLayerWrapper(attn_layer, rotary_emb)
attn_wrapper.eval()

seq_len = 8
hidden_dim = 2048
hidden_dummy  = torch.randn(1, seq_len, hidden_dim)
attn_mask_dummy = torch.zeros(1, 1, seq_len, seq_len)   # causal mask (all 0 = attend all)
pos_ids_dummy = torch.arange(seq_len).unsqueeze(0)      # (1, seq_len)

with torch.no_grad():
    torch.onnx.export(
        attn_wrapper,
        (hidden_dummy, attn_mask_dummy, pos_ids_dummy),
        str(OUT_DIR / "attn_layer.onnx"),
        input_names=["hidden_states", "attention_mask", "position_ids"],
        output_names=["output_hidden_states"],
        dynamic_axes={
            "hidden_states":       {1: "seq_len"},
            "attention_mask":      {2: "seq_len", 3: "seq_len"},
            "position_ids":        {1: "seq_len"},
            "output_hidden_states":{1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"  Saved → {OUT_DIR}/attn_layer.onnx")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT 3: Linear-attention decoder layer (Qwen3_5GatedDeltaNet + MLP)
# Layers 0,1,2, 4,5,6, ... — recurrent linear attention (no KV cache)
# Input:  hidden_states (seq_len, hidden_dim=2048)
# Output: hidden_states (seq_len, 2048)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Exporting linear-attention decoder layer (layer 0)...")

class LinearAttnLayerWrapper(nn.Module):
    """Wraps one GatedDeltaNet decoder layer. No RoPE, no cache → clean static graph."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, attention_mask):
        # GatedDeltaNet (linear_attention) does not use position_embeddings,
        # but DecoderLayer.forward() still requires it — pass dummy (None, None) tuple.
        out = self.layer(
            hidden_states=hidden_states,
            position_embeddings=(None, None),
            attention_mask=attention_mask,
            past_key_values=None,
            cache_position=None,
        )
        return out  # (batch, seq_len, hidden_dim)

linear_layer = model.model.language_model.layers[0]  # layer 0 = first GatedDeltaNet layer
linear_wrapper = LinearAttnLayerWrapper(linear_layer)
linear_wrapper.eval()

# GatedDeltaNet attention_mask is a flat mask (1D per token), not 4D causal mask
lin_mask_dummy = torch.ones(1, seq_len, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        linear_wrapper,
        (hidden_dummy, lin_mask_dummy),
        str(OUT_DIR / "linear_attn_layer.onnx"),
        input_names=["hidden_states", "attention_mask"],
        output_names=["output_hidden_states"],
        dynamic_axes={
            "hidden_states":       {1: "seq_len"},
            "attention_mask":      {1: "seq_len"},
            "output_hidden_states":{1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
print(f"  Saved → {OUT_DIR}/linear_attn_layer.onnx")

print(f"""
══════════════════════════════════════════════════════
  Export complete. Files saved to {OUT_DIR}/

  Open each in https://netron.app/ :
  ┌──────────────────────────┬────────────────────────────────────────┐
  │ File                     │ What you'll see                        │
  ├──────────────────────────┼────────────────────────────────────────┤
  │ vision_encoder.onnx      │ Conv3d patches → 24 blocks → merger    │
  │ attn_layer.onnx          │ GQA attention (q/k/v proj, RoPE, MLP)  │
  │ linear_attn_layer.onnx   │ GatedDeltaNet (conv1d, gates, MLP)     │
  └──────────────────────────┴────────────────────────────────────────┘
══════════════════════════════════════════════════════
""")


'''
  if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
  Saved → /home/onnx_exports/linear_attn_layer.onnx

══════════════════════════════════════════════════════
  Export complete. Files saved to /home/onnx_exports/

  Open each in https://netron.app/ :
  ┌──────────────────────────┬────────────────────────────────────────┐
  │ File                     │ What you'll see                        │
  ├──────────────────────────┼────────────────────────────────────────┤
  │ vision_encoder.onnx      │ Conv3d patches → 24 blocks → merger    │
  │ attn_layer.onnx          │ GQA attention (q/k/v proj, RoPE, MLP)  │
  │ linear_attn_layer.onnx   │ GatedDeltaNet (conv1d, gates, MLP)     │
  └──────────────────────────┴────────────────────────────────────────┘
'''