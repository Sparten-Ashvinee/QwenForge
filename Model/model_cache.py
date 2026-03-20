import time
import copy
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", device_map="cuda")

IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
QUESTIONS = [
    "What animal is on the candy?",
    "What color is the candy wrapper?",
]

def clone_kv_cache(cache):
    """Deep-copy the model-specific cache (Qwen3_5DynamicCache) so each generate()
    call gets its own copy. Uses deepcopy to handle all cache types (KV, conv_states,
    recurrent_states) without depending on internal attribute names."""
    return copy.deepcopy(cache)

# ── Step 1: Build the image prefix and cache its KV once ─────────────────────
# Only the image content is passed (no question text yet).
# add_generation_prompt=False so we don't append the assistant token yet.
prefix_messages = [{"role": "user", "content": [{"type": "image", "url": IMAGE_URL}]}]
prefix_inputs = processor.apply_chat_template(
    prefix_messages,
    add_generation_prompt=False,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    # Forward pass only (no generation) — populates KV cache for all image tokens
    prefix_out = model(**prefix_inputs, use_cache=True)
torch.cuda.synchronize()
prefix_time = (time.perf_counter() - t0) * 1000

prefix_kv  = prefix_out.past_key_values   # KV tensors for all ~11,890 image tokens
prefix_len = prefix_inputs["input_ids"].shape[1]
print(f"Image prefix cached: {prefix_len} tokens  ({prefix_time:.0f} ms — paid once)")
print("─" * 55)

# ── Step 2: Ask multiple questions reusing the cached image prefix ─────────────
for question in QUESTIONS:
    # Tokenize only the question text + assistant generation prompt (no image)
    suffix_ids = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": question}]}],
        add_generation_prompt=True,    # appends <|im_start|>assistant\n
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)                 # shape: (1, suffix_len)
    suffix_len = suffix_ids.shape[1]

    # Attention mask must span the full sequence (prefix + suffix) so positional
    # offsets are computed correctly even though we only pass the suffix tokens.
    full_attn_mask = torch.ones(1, prefix_len + suffix_len, dtype=torch.long, device=model.device)

    # Clone the prefix KV so model.generate() doesn't mutate the shared cache.
    kv_for_this_question = clone_kv_cache(prefix_kv)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=suffix_ids,              # only new tokens; image is already in KV cache
            attention_mask=full_attn_mask,     # full length for correct position IDs
            past_key_values=kv_for_this_question,  # reuse image KV — skips image prefill
            max_new_tokens=40,
        )
    torch.cuda.synchronize()
    q_time = (time.perf_counter() - t0) * 1000

    answer = processor.decode(outputs[0][suffix_len:], skip_special_tokens=True)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"   {q_time:.0f} ms  (image prefill skipped for this question)")
    print()