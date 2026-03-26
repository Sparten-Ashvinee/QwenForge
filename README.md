# QwenForge

A deep structural study of **Qwen3.5-2B** — a hybrid multimodal model combining a Vision Transformer encoder with a language model that interleaves linear (GatedDeltaNet) and full (GQA) attention layers.

This repository dissects the model architecture end-to-end: from raw pixel inputs through the vision encoder and token merging, into the hybrid decoder, and all the way to autoregressive generation. It also covers GPU performance profiling, roofline analysis, KV-cache optimization, and ONNX export for architecture visualization.

---

## Model

**[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)** — 2B parameter vision-language model.

<details>
<summary>Architecture summary</summary>

```
Qwen3_5ForConditionalGeneration
├── Vision Encoder  (Qwen3_5VisionModel)
│   ├── patch_embed     Conv3d(3, 1024, kernel=(2,16,16))  — spatio-temporal patch extraction
│   ├── pos_embed       Embedding(2304, 1024)
│   ├── rotary_pos_emb  Qwen3_5VisionRotaryEmbedding
│   ├── blocks          24 × VisionBlock
│   │   ├── LayerNorm → Attention(QKV: 1024→3072) → proj(1024→1024)
│   │   └── LayerNorm → MLP(1024→4096→1024, GELUTanh)
│   └── merger          PatchMerger: (N,1024) → (N/4, 2048)
│                         concat 2×2 neighbours → fc1(4096→4096, GELU) → fc2(4096→2048)
│
└── Language Model  (Qwen3_5TextModel)
    ├── embed_tokens    Embedding(248320, 2048)
    └── layers          28 × DecoderLayer  (hybrid attention)
        ├── Layers 0-2, 4-6, 8-10 ...  GatedDeltaNet  (linear attention)
        │   Conv1d(groups) + RMSNormGated + in_proj(QKV/z/b/a) + out_proj
        └── Layers 3, 7, 11, 15, 19, 23, 27  Qwen3_5Attention  (full GQA: 16Q / 2KV heads)
            + MLP(2048→6144→2048, SiLU)
```

</details>

---

## Setup

### Requirements
- Python 3.10+
- CUDA 12.4 + compatible GPU (scripts default to A100; adjust constants for other GPUs)
- PyTorch 2.6.0 with CUDA 12.4 wheel

### Installation

```bash
# 1. Install PyTorch with CUDA 12.4 support
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements.txt
```

### Running Scripts

```bash
# Baseline inference
python Model/quenllm_original.py

# End-to-end tensor shape trace
python e2e_flow/model_e2e.py

# Latency + VRAM profiling
python Model/model_ttft_calc.py

# Roofline analysis (saves plot to output/)
python Model/model_performance.py
python Model/model3_roofline.py

# KV prefix caching demo
python Model/model_cache.py

# GPU-optimized inference
python gpu_optimization/model_gpu.PY

# ONNX export (outputs to /home/onnx_exports/ by default)
python export_model/export_onnx.PY
```

---

## Scripts

### `Model/quenllm_original.py` — Baseline Inference
Minimal script for loading the model from HuggingFace and running a multimodal prompt (image URL + text question). Entry point for verifying the environment.

### `e2e_flow/model_e2e.py` — End-to-End Tensor Trace
Walks the full data path from raw inputs to the merged sequence, printing every tensor shape at each stage:
- **Stage 0** — Raw PIL image + text string
- **Stage 1** — Processor output: `pixel_values`, `input_ids`, `image_grid_thw`
- **Stage 2** — Vision encoder: `patch_embed` → `pos_embed` → 24 VisionBlocks → `merger` → `(N_img_tokens, 2048)`
- **Stage 3** — Text token embedding: `embed_tokens(248320×2048)`
- **Stage 4** — Merged sequence (image tokens injected into text positions)

### `Model/model_ttft_calc.py` — Latency & VRAM Metrics
Measures and reports:
| Metric | Description |
|--------|-------------|
| **TTFT** | Time To First Token (ms) |
| **ITL** | Inter-Token Latency (ms/token) |
| **Throughput** | Tokens/second |
| **VRAM weights** | `Σ(params × bytes)` |
| **VRAM gradients** | Training reference (0 at inference) |
| **VRAM optimizer** | AdamW 8 bytes/param estimate |
| **VRAM peak** | `torch.cuda.max_memory_allocated()` |

Uses `TextIteratorStreamer` with a background thread so per-token timestamps are captured without blocking.

### `gpu_optimization/model_gpu.PY` — GPU-Optimized Inference
Applies five optimization techniques on top of the baseline:

| Technique | Effect |
|-----------|--------|
| `torch.bfloat16` | Halves weight memory; enables Tensor Core utilization |
| Flash attention (`enable_flash_sdp`) | Fused O(N) memory attention — faster and lower VRAM than default O(N²) |
| TF32 matmuls | 19-bit mantissa on Ampere+ — same throughput as fp16 with fp32 range |
| CUDA high-priority stream | Decode kernels bypass default-stream queuing |
| Processor `max_pixels` cap | Limits image patch tokens (~512 vs ~11,851 default) for prefill speed |

> **Note:** `torch.compile` is intentionally disabled. The hybrid `Qwen3_5DynamicCache` with `_update_linear_attn_mask()` is not yet supported by Dynamo's tracer. The remaining four optimizations are fully active.

### `export_model/export_onnx.PY` — ONNX Export
Exports three self-contained subgraphs for architecture visualization in [Netron](https://netron.app):

| File | Inputs | Outputs | Notes |
|------|--------|---------|-------|
| `vision_encoder.onnx` | `pixel_values (N,1536)`, `grid_thw (1,3)` | `image_features (N/4, 2048)` | Dynamic `N_patches` axis |
| `attn_layer.onnx` | `hidden_states`, `attention_mask`, `position_ids` | `hidden_states (B,S,2048)` | Layer 3 — first full GQA layer |
| `linear_attn_layer.onnx` | `hidden_states (S,2048)` | `hidden_states (S,2048)` | GatedDeltaNet layer |

The full model cannot be traced end-to-end with `jit.trace` due to data-dependent control flow in the hybrid cache. Uses `opset_version=17` with the legacy `jit.trace` exporter (`dynamo=False`) to avoid failures on `torch.linspace` in `fast_pos_embed_interpolate`.

---

## Key Implementation Notes

- **FLOPs estimation**: `FlopCounterMode` is incompatible with GQA (Q heads ≠ KV heads) in this model. All scripts use the standard transformer approximation: `FLOPs ≈ 2 × num_params × seq_len`.
- **KV cache type**: Qwen3.5-2B uses `Qwen3_5DynamicCache` — a hybrid cache that stores both standard KV tensors (full attention layers) and conv/recurrent states (linear attention layers). `copy.deepcopy` is required to clone it correctly.
- **ONNX export precision**: Must run in `fp32` (`dtype=torch.float32`) as the legacy `jit.trace` ONNX exporter does not support mixed-precision tracing.
- **`attn_implementation="eager"`**: Required for ONNX export — `F.scaled_dot_product_attention` with `enable_gqa=True` is not convertible by the legacy exporter.

---

## References

- [Qwen/Qwen3.5-2B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Netron — ONNX model visualizer](https://netron.app)
- [Roofline Model (Williams et al., 2008)](https://dl.acm.org/doi/10.1145/1498765.1498785)
