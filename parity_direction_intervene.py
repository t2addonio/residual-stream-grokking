#!/usr/bin/env python3
"""
parity_direction_intervene.py

Isolate a residual-stream direction in phi-2 that carries sparse-parity
information, then phase-cancel it and measure the effect.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import random
import numpy as np

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "microsoft/phi-2"
LAYER = 20                    # late layer – we can sweep later
COEFF = 2.0                   # same working coefficient from the original work
N_BITS = 12                   # smaller than 20 so it fits context comfortably
K_SPARSE = 4                  # same as original
N_TRAIN = 256                 # examples for direction extraction
N_TEST  = 128                 # examples for measuring effect
DEVICE = "cpu"                # Intel Mac
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
print("Model loaded.")

# ------------------------------------------------------------------
# Sparse parity helpers
# ------------------------------------------------------------------
def make_parity_example(n_bits=N_BITS, k=K_SPARSE):
    bits = [random.randint(0, 1) for _ in range(n_bits)]
    label = sum(bits[:k]) % 2
    # Simple text format the model can handle
    bit_str = " ".join(str(b) for b in bits)
    prompt = f"Bits: {bit_str}\nParity of first {k} bits:"
    return prompt, label, bits

def make_dataset(n, seed=0):
    random.seed(seed)
    data = [make_parity_example() for _ in range(n)]
    return data

# ------------------------------------------------------------------
# Residual collection
# ------------------------------------------------------------------
@torch.no_grad()
def get_residual(prompt: str, layer: int = LAYER):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] = embed, hidden_states[i+1] = after layer i
    resid = outputs.hidden_states[layer + 1]          # [1, seq, d]
    # last token residual
    last = resid[0, -1, :].cpu()
    return last

@torch.no_grad()
def get_residuals_for_dataset(dataset, layer=LAYER):
    res_0, res_1 = [], []
    labels = []
    for prompt, label, _ in dataset:
        r = get_residual(prompt, layer=layer)
        if label == 0:
            res_0.append(r)
        else:
            res_1.append(r)
        labels.append(label)
    res_0 = torch.stack(res_0) if res_0 else torch.empty(0)
    res_1 = torch.stack(res_1) if res_1 else torch.empty(0)
    return res_0, res_1, labels

# ------------------------------------------------------------------
# Direction isolation (difference of means + optional probe)
# ------------------------------------------------------------------
def extract_parity_direction(res_0, res_1):
    """Simple difference-of-means direction (normalized)."""
    mu0 = res_0.mean(dim=0)
    mu1 = res_1.mean(dim=0)
    d = mu1 - mu0
    d = d / (d.norm() + 1e-8)
    return d

def extract_parity_subspace(res_0, res_1, k=4):
    """Low-rank version via SVD on the contrast + data."""
    mu0 = res_0.mean(dim=0)
    mu1 = res_1.mean(dim=0)
    diff = (mu1 - mu0).unsqueeze(0)
    # center class 1 residuals
    centered = res_1 - mu1
    mat = torch.cat([diff, centered], dim=0)
    _, _, Vh = torch.linalg.svd(mat.float(), full_matrices=False)
    B = Vh[:k]
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B

# ------------------------------------------------------------------
# Phase-cancellation hook
# ------------------------------------------------------------------
def make_cancel_hook(direction_or_B, coeff=COEFF, is_subspace=False):
    direction_or_B = direction_or_B.to(DEVICE)
    def hook(module, input, output):
        resid = output[0] if isinstance(output, tuple) else output
        if is_subspace:
            # B is [k, d]
            proj = torch.einsum("bsd,kd->bsk", resid, direction_or_B)
            delta = torch.einsum("bsk,kd->bsd", proj, direction_or_B)
        else:
            # single direction
            # proj scalar per position
            scalar = torch.einsum("bsd,d->bs", resid, direction_or_B)
            delta = scalar.unsqueeze(-1) * direction_or_B
        resid = resid - coeff * delta
        if isinstance(output, tuple):
            return (resid,) + output[1:]
        return resid
    return hook

# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_parity(dataset, hook_handle=None):
    """Return accuracy and average correct-logit margin."""
    correct = 0
    margins = []
    for prompt, label, _ in dataset:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]          # last position

        # We look at the logits for tokens "0" and "1"
        tok0 = tokenizer.encode("0", add_special_tokens=False)[0]
        tok1 = tokenizer.encode("1", add_special_tokens=False)[0]
        logit0 = logits[tok0].item()
        logit1 = logits[tok1].item()

        pred = 1 if logit1 > logit0 else 0
        if pred == label:
            correct += 1
        # margin toward the correct answer
        margin = (logit1 - logit0) if label == 1 else (logit0 - logit1)
        margins.append(margin)

    acc = correct / len(dataset)
    avg_margin = float(np.mean(margins))
    return acc, avg_margin

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
print("Building datasets...")
train_data = make_dataset(N_TRAIN, seed=42)
test_data  = make_dataset(N_TEST,  seed=123)

print("Collecting residuals (this takes a few minutes)...")
res_0, res_1, _ = get_residuals_for_dataset(train_data, layer=LAYER)
print(f"  class 0: {res_0.shape[0]} examples, class 1: {res_1.shape[0]} examples")

print("Extracting parity direction...")
direction = extract_parity_direction(res_0, res_1)
B = extract_parity_subspace(res_0, res_1, k=4)
print(f"  direction norm: {direction.norm():.4f}")
print(f"  subspace shape: {B.shape}")

# Baseline performance
print("\n=== BASELINE (no intervention) ===")
acc_base, margin_base = evaluate_parity(test_data)
print(f"Accuracy: {acc_base:.3f}  |  Avg correct margin: {margin_base:.3f}")

# Phase-cancel the single direction
print("\n=== PHASE-CANCEL single direction (coeff=2.0) ===")
handle = model.model.layers[LAYER].register_forward_hook(
    make_cancel_hook(direction, coeff=COEFF, is_subspace=False)
)
acc_dir, margin_dir = evaluate_parity(test_data)
handle.remove()
print(f"Accuracy: {acc_dir:.3f}  |  Avg correct margin: {margin_dir:.3f}")

# Phase-cancel the small subspace
print("\n=== PHASE-CANCEL 4-dim subspace (coeff=2.0) ===")
handle = model.model.layers[LAYER].register_forward_hook(
    make_cancel_hook(B, coeff=COEFF, is_subspace=True)
)
acc_sub, margin_sub = evaluate_parity(test_data)
handle.remove()
print(f"Accuracy: {acc_sub:.3f}  |  Avg correct margin: {margin_sub:.3f}")

print("\nDone.")
print("Interpretation guide:")
print("  - If accuracy / margin drops a lot under cancellation → we found a load-bearing parity direction.")
print("  - If almost no change → the parity information is more distributed or in a different layer.")
