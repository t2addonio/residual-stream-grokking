#!/usr/bin/env python3
"""
mechanistic_depth_analysis.py

Goal: Increase mechanistic depth of the memory subspace.
1. Project residual-stream activations onto the extracted memory directions.
2. Rank examples by how strongly they activate the subspace.
3. Measure how much of the final logits is carried by the subspace (logit-lens style).
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
import json
from tqdm import tqdm

# --------------------------------------------------
# Config
# --------------------------------------------------
P = 113
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 9304          # pick a seed that grokked cleanly
ALPHA = 0.35
RESULTS_DIR = Path("results_mech")
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Model & Data helpers
# --------------------------------------------------
def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        act_fn="relu",
        normalization_type=None,
        d_vocab=P,
        n_ctx=2,                    # a, b, =
        seed=SEED,
    )
    model = HookedTransformer(cfg).to(DEVICE)
    return model

def make_dataset(p=P):
    a = torch.arange(p, device=DEVICE)
    b = torch.arange(p, device=DEVICE)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    xx = torch.stack([aa.flatten(), bb.flatten()], dim=1)   # (p², 2)
    yy = (aa + bb) % p
    return xx, yy.flatten()

def split_dataset(x, y, seed=SEED, train_frac=0.5):
    rng = np.random.RandomState(seed)
    n = len(x)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])

# --------------------------------------------------
# Memory subspace extraction (same method as before)
# --------------------------------------------------
@torch.no_grad()
def extract_mem_subspace(model, train_x, train_y, test_x, test_y, k=8):
    def get_resid(x):
        tokens = torch.stack([
            x[:, 0],
            x[:, 1],
            torch.full((x.shape[0],), P-1, device=DEVICE)   # dummy "=" token if needed
        ], dim=1) if False else x   # adjust if your tokenization differs
        # Simpler: just use the two operands + result position
        logits, cache = model.run_with_cache(x)
        return cache["resid_post", -1][:, -1, :]   # final position

    # For modular addition the usual tokenization is [a, b] → predict a+b
    # Adjust the call below to match how you actually feed the model
    resid_train = []
    resid_test  = []

    # Batch for memory
    bs = 512
    for i in range(0, len(train_x), bs):
        xb = train_x[i:i+bs]
        _, cache = model.run_with_cache(xb)
        resid_train.append(cache["resid_post", -1][:, -1, :])
    for i in range(0, len(test_x), bs):
        xb = test_x[i:i+bs]
        _, cache = model.run_with_cache(xb)
        resid_test.append(cache["resid_post", -1][:, -1, :])

    resid_train = torch.cat(resid_train, dim=0)
    resid_test  = torch.cat(resid_test,  dim=0)

    # Contrast
    mean_train = resid_train.mean(0)
    mean_test  = resid_test.mean(0)
    contrast   = (mean_train - mean_test).unsqueeze(0)

    # SVD
    U, S, Vh = torch.linalg.svd(contrast, full_matrices=False)
    B = Vh[:k]          # (k, d_model)
    return B, resid_train, resid_test

# --------------------------------------------------
# Analysis 1: Projection magnitude ranking
# --------------------------------------------------
@torch.no_grad()
def projection_analysis(model, B, x, y, top_n=20):
    """Rank examples by ||proj_B(resid)||"""
    _, cache = model.run_with_cache(x)
    resid = cache["resid_post", -1][:, -1, :]          # (N, d)
    proj  = (resid @ B.T) @ B                          # (N, d)
    norms = proj.norm(dim=-1)                          # (N,)

    order = norms.argsort(descending=True)
    results = []
    for i in order[:top_n]:
        a, b = x[i].tolist()
        results.append({
            "a": a,
            "b": b,
            "sum_mod_p": int(y[i]),
            "proj_norm": float(norms[i]),
        })
    return results, norms

# --------------------------------------------------
# Analysis 2: Subspace contribution to logits
# --------------------------------------------------
@torch.no_grad()
def subspace_logit_contribution(model, B, x, y):
    """
    Measure how much of the correct-logit is carried by the
    component of the residual stream that lies in span(B).
    """
    _, cache = model.run_with_cache(x)
    resid = cache["resid_post", -1][:, -1, :]          # (N, d)

    # Full residual → logits
    logits_full = model.unembed(resid)                 # (N, p)

    # Projection onto memory subspace
    proj = (resid @ B.T) @ B
    logits_mem = model.unembed(proj)

    # Orthogonal complement
    resid_orth = resid - proj
    logits_orth = model.unembed(resid_orth)

    # Correct logit values
    correct = y
    full_correct  = logits_full[torch.arange(len(y)), correct]
    mem_correct   = logits_mem[torch.arange(len(y)), correct]
    orth_correct  = logits_orth[torch.arange(len(y)), correct]

    return {
        "mean_full_correct_logit": float(full_correct.mean()),
        "mean_mem_correct_logit":  float(mem_correct.mean()),
        "mean_orth_correct_logit": float(orth_correct.mean()),
        "frac_carried_by_mem":     float((mem_correct / (full_correct + 1e-8)).mean()),
    }

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Building model...")
    model = make_model()

    # Load a grokked checkpoint
    checkpoint_path = "grokked_seed9304.pt"   # ← change this for each seed
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y)

    print("Extracting memory subspace...")
    B, resid_train, resid_test = extract_mem_subspace(model, train_x, train_y, test_x, test_y, k=8)
    print(f"Subspace shape: {B.shape}")

    print("\n=== Projection ranking (test set) ===")
    top_examples, norms = projection_analysis(model, B, test_x, test_y, top_n=15)
    for i, ex in enumerate(top_examples):
        print(f"{i+1:2d}. a={ex['a']:3d} b={ex['b']:3d} → {ex['sum_mod_p']:3d}  |proj|={ex['proj_norm']:.3f}")

    print("\n=== Subspace contribution to correct logits ===")
    contrib = subspace_logit_contribution(model, B, test_x[:2048], test_y[:2048])
    for k, v in contrib.items():
        print(f"  {k}: {v:.4f}")

    # Save
    out = {
        "seed": SEED,
        "top_examples": top_examples,
        "logit_contribution": contrib,
    }
    out_path = RESULTS_DIR / f"mech_depth_seed{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
