#!/usr/bin/env python3
"""
parity_direction_sweep.py

- Strong few-shot prompts so the model can actually solve sparse parity
- Extract parity direction via difference-of-means
- Phase-cancel at multiple late layers and measure the effect
"""

import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "microsoft/phi-2"
LAYERS_TO_TEST = [20, 24, 28, 31]     # late layers
COEFF = 2.0
N_BITS = 10                           # slightly smaller for cleaner few-shot
K_SPARSE = 3
N_TRAIN = 192
N_TEST  = 96
DEVICE = "cpu"
DTYPE = torch.float32

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=None,
    trust_remote_code=True,
).to(DEVICE)
model.eval()
print("Model loaded.\n")

# ------------------------------------------------------------------
# Few-shot sparse parity
# ------------------------------------------------------------------
FEW_SHOT = """Here are examples of parity of the first 3 bits:

Bits: 1 0 1 0 1 1 0 0 1 0
Parity of first 3 bits: 0

Bits: 1 1 0 1 0 0 1 1 0 1
Parity of first 3 bits: 0

Bits: 0 1 1 0 1 0 0 1 1 0
Parity of first 3 bits: 0

Bits: 1 0 0 1 1 0 1 0 0 1
Parity of first 3 bits: 1

Bits: 0 0 1 1 0 1 1 0 1 0
Parity of first 3 bits: 1

Bits: 1 1 1 0 0 1 0 1 0 1
Parity of first 3 bits: 1

Now solve this one:
"""

def make_parity_example(n_bits=N_BITS, k=K_SPARSE):
    bits = [random.randint(0, 1) for _ in range(n_bits)]
    label = sum(bits[:k]) % 2
    bit_str = " ".join(str(b) for b in bits)
    prompt = FEW_SHOT + f"Bits: {bit_str}\nParity of first {k} bits:"
    return prompt, label

def make_dataset(n, seed):
    random.seed(seed)
    return [make_parity_example() for _ in range(n)]

# ------------------------------------------------------------------
# Residual + direction helpers
# ------------------------------------------------------------------
@torch.no_grad()
def get_residual(prompt, layer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    resid = outputs.hidden_states[layer + 1][0, -1, :].cpu()
    return resid

def collect_residuals(dataset, layer):
    res0, res1 = [], []
    for prompt, label in dataset:
        r = get_residual(prompt, layer)
        (res0 if label == 0 else res1).append(r)
    res0 = torch.stack(res0) if res0 else torch.empty(0, 2560)
    res1 = torch.stack(res1) if res1 else torch.empty(0, 2560)
    return res0, res1

def extract_direction(res0, res1):
    d = res1.mean(0) - res0.mean(0)
    return d / (d.norm() + 1e-8)

def make_cancel_hook(direction, coeff=COEFF):
    direction = direction.to(DEVICE)
    def hook(module, input, output):
        resid = output[0] if isinstance(output, tuple) else output
        scalar = torch.einsum("bsd,d->bs", resid, direction)
        delta = scalar.unsqueeze(-1) * direction
        resid = resid - coeff * delta
        if isinstance(output, tuple):
            return (resid,) + output[1:]
        return resid
    return hook

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(dataset):
    correct = 0
    margins = []
    tok0 = tokenizer.encode("0", add_special_tokens=False)[0]
    tok1 = tokenizer.encode("1", add_special_tokens=False)[0]

    for prompt, label in dataset:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        logits = model(**inputs).logits[0, -1, :]
        logit0 = logits[tok0].item()
        logit1 = logits[tok1].item()
        pred = 1 if logit1 > logit0 else 0
        if pred == label:
            correct += 1
        margin = (logit1 - logit0) if label == 1 else (logit0 - logit1)
        margins.append(margin)

    acc = correct / len(dataset)
    return acc, float(np.mean(margins))

# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
print("Building datasets...")
train_data = make_dataset(N_TRAIN, seed=42)
test_data  = make_dataset(N_TEST,  seed=123)

print(f"\n{'Layer':<8} {'Base Acc':>10} {'Base Margin':>12} {'Cancel Acc':>12} {'Cancel Margin':>14} {'Δ Acc':>8}")
print("-" * 70)

for layer in LAYERS_TO_TEST:
    print(f"\n>>> Working on layer {layer} ...")

    # Collect residuals and extract direction at this layer
    res0, res1 = collect_residuals(train_data, layer)
    direction = extract_direction(res0, res1)

    # Baseline
    acc_base, margin_base = evaluate(test_data)

    # Phase-cancel
    handle = model.model.layers[layer].register_forward_hook(
        make_cancel_hook(direction, coeff=COEFF)
    )
    acc_cancel, margin_cancel = evaluate(test_data)
    handle.remove()

    delta = acc_cancel - acc_base
    print(f"{layer:<8} {acc_base:>10.3f} {margin_base:>12.3f} {acc_cancel:>12.3f} {margin_cancel:>14.3f} {delta:>+8.3f}")

print("\nDone.")
print("Look for layers where Base Acc is clearly above chance (~0.50) and Cancel Acc drops.")
