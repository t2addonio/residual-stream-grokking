#!/usr/bin/env python3
"""
Dynamic-Range Causal Experiment (v1)
Tests whether relative dynamic range of mem vs gen residual-stream components
causally controls the grokking transition.

Usage examples (run on different GPUs):
  python run_dynamic_range.py --condition control --start 6000 --end 6014
  python run_dynamic_range.py --condition mem_suppress --start 6100 --end 6114 --alpha 0.5
  python run_dynamic_range.py --condition gen_amplify --start 6200 --end 6214 --beta 1.8
  python run_dynamic_range.py --condition isotropic --start 6300 --end 6314 --alpha 0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import random
import numpy as np
import json
import os
import argparse
from datetime import datetime
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
p = 113
D_MODEL = 128
LAYER = 1          # residual stream after this layer
K_DIMS = 12        # number of directions to keep for each subspace
EXTRACT_STEP = 2000
INTERVENE_STEP = 2000

def get_model():
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=D_MODEL,
        d_head=32,
        n_heads=4,
        d_mlp=512,
        n_ctx=3,
        d_vocab=p + 1,
        act_fn="relu",
        positional_embedding_type="standard",
    )
    return HookedTransformer(cfg).to(device)

def make_data(batch_size=512, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    a = torch.randint(0, p, (batch_size, 1), device=device)
    b = torch.randint(0, p, (batch_size, 1), device=device)
    x = torch.cat([a, b, torch.zeros_like(a)], dim=1)
    y = (a + b) % p
    return x, y.squeeze(-1)

# ============================================================
# Subspace extraction (simple contrast SVD)
# ============================================================
@torch.no_grad()
def extract_subspaces(model, n_samples=2048):
    model.eval()
    # Training examples (memorized)
    x_train, _ = make_data(n_samples, seed=123)
    _, cache_train = model.run_with_cache(x_train)
    resid_train = cache_train[f"blocks.{LAYER}.hook_resid_post"].reshape(-1, D_MODEL)

    # Held-out examples
    x_held, _ = make_data(n_samples, seed=999)
    _, cache_held = model.run_with_cache(x_held)
    resid_held = cache_held[f"blocks.{LAYER}.hook_resid_post"].reshape(-1, D_MODEL)

    # Contrast: directions that are stronger on train than held-out
    mean_train = resid_train.mean(dim=0)
    mean_held = resid_held.mean(dim=0)
    contrast = mean_train - mean_held

    # Also do a quick SVD on the train activations for richer basis
    U, S, V = torch.svd_lowrank(resid_train - resid_train.mean(0), q=K_DIMS)
    mem_basis = V[:, :K_DIMS].T  # (K, d_model)

    # Simple gen proxy: orthogonal complement directions (or top of held-out)
    U_h, S_h, V_h = torch.svd_lowrank(resid_held - resid_held.mean(0), q=K_DIMS)
    gen_basis = V_h[:, :K_DIMS].T

    model.train()
    return mem_basis.to(device), gen_basis.to(device)

# ============================================================
# Dynamic range measurement
# ============================================================
@torch.no_grad()
def compute_ranges(model, mem_basis, gen_basis, n_samples=512):
    model.eval()
    x, _ = make_data(n_samples)
    _, cache = model.run_with_cache(x)
    resid = cache[f"blocks.{LAYER}.hook_resid_post"].reshape(-1, D_MODEL)

    # Project and take RMS
    proj_mem = resid @ mem_basis.T
    proj_gen = resid @ gen_basis.T
    range_mem = proj_mem.pow(2).mean().sqrt().item()
    range_gen = proj_gen.pow(2).mean().sqrt().item()
    model.train()
    return range_mem, range_gen

# ============================================================
# Intervention hooks
# ============================================================
def make_intervention_hook(mem_basis, gen_basis, condition, alpha=0.5, beta=1.8):
    """
    Returns a hook function that modifies the residual stream.
    """
    mem_basis = mem_basis.detach()
    gen_basis = gen_basis.detach()

    def hook_fn(resid, hook):
        # resid shape: [batch, pos, d_model]
        original_shape = resid.shape
        flat = resid.reshape(-1, D_MODEL)

        if condition == "mem_suppress":
            # Project out (or scale down) mem subspace
            proj = (flat @ mem_basis.T) @ mem_basis
            flat = flat - (1.0 - alpha) * proj

        elif condition == "gen_amplify":
            proj = (flat @ gen_basis.T) @ gen_basis
            flat = flat + (beta - 1.0) * proj

        elif condition == "isotropic":
            # Matched total norm change without preferred direction
            # Simple version: scale entire residual by a factor that approximates
            # the average effect of the directional interventions
            scale = 0.85 if alpha < 1.0 else 1.15
            flat = flat * scale

        # Control does nothing
        return flat.reshape(original_shape)

    return hook_fn

# ============================================================
# Main training loop for one seed
# ============================================================
def run_seed(seed, condition, alpha=0.5, beta=1.8, max_steps=20001):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    mem_basis = None
    gen_basis = None
    hook_handle = None

    history = {
        "steps": [],
        "train_loss": [],
        "test_loss": [],
        "range_mem": [],
        "range_gen": [],
        "ratio": [],
    }

    print(f"\n=== {condition.upper()} | Seed {seed} | alpha={alpha} beta={beta} ===")

    for step in tqdm(range(max_steps)):
        # ---- training step ----
        x, y = make_data()
        logits = model(x)
        loss = criterion(logits[:, -1], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- extract subspaces once ----
        if step == EXTRACT_STEP:
            mem_basis, gen_basis = extract_subspaces(model)
            print(f"  Subspaces extracted at step {step}")

            if condition != "control":
                hook_fn = make_intervention_hook(mem_basis, gen_basis, condition, alpha, beta)
                hook_handle = model.add_hook(f"blocks.{LAYER}.hook_resid_post", hook_fn)
                print(f"  Intervention hook installed ({condition})")

        # ---- logging ----
        if step % 500 == 0 or step == max_steps - 1:
            # quick test loss
            with torch.no_grad():
                x_t, y_t = make_data(1024, seed=42)
                test_logits = model(x_t)
                test_loss = criterion(test_logits[:, -1], y_t).item()

            range_mem, range_gen = 0.0, 0.0
            ratio = 0.0
            if mem_basis is not None:
                range_mem, range_gen = compute_ranges(model, mem_basis, gen_basis)
                ratio = range_gen / (range_mem + 1e-8)

            history["steps"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)
            history["range_mem"].append(range_mem)
            history["range_gen"].append(range_gen)
            history["ratio"].append(ratio)

            if step % 2000 == 0:
                print(f"Step {step:5d} | train {loss.item():.4f} | test {test_loss:.4f} | "
                      f"R_mem {range_mem:.3f} | R_gen {range_gen:.3f} | ratio {ratio:.3f}")

    # cleanup
    if hook_handle is not None:
        hook_handle.remove()

    final_test = history["test_loss"][-1]
    grokked = final_test < 0.8
    print(f"Final test loss = {final_test:.4f} | Grokked: {grokked}")

    return {
        "condition": condition,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "final_test_loss": final_test,
        "grokked": grokked,
        "history": history,
    }

# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True,
                        choices=["control", "mem_suppress", "gen_amplify", "isotropic"])
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Mem suppress factor")
    parser.add_argument("--beta", type=float, default=1.8, help="Gen amplify factor")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    results = []

    for seed in range(args.start, args.end + 1):
        res = run_seed(seed, args.condition, alpha=args.alpha, beta=args.beta)
        results.append(res)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"results/{args.condition}_{args.start}_{args.end}_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    successes = sum(1 for r in results if r["grokked"])
    print("\n" + "=" * 60)
    print(f"{args.condition}: {successes}/{len(results)} grokked ({successes/len(results):.1%})")
    print(f"Saved → {fname}")

if __name__ == "__main__":
    main()
