#!/usr/bin/env python3
"""
fourier_overlap_test.py

Test whether the extracted "memory" subspace overlaps with
classic Nanda-style Fourier features for modular addition,
and whether they are causally distinct.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
import json
from datetime import datetime

# ───────────────────────── Config ─────────────────────────
P = 113
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
K_DIMS = 16
LAYER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_fourier")
RESULTS_DIR.mkdir(exist_ok=True)

# ───────────────────────── Model & Data ─────────────────────────
def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        act_fn="relu",
        normalization_type=None,
        d_vocab=P,
        n_ctx=2,
        device=DEVICE,
    )
    return HookedTransformer(cfg)

def make_dataset(p=P):
    a = torch.arange(p)
    b = torch.arange(p)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    y = ((aa + bb) % p).flatten()
    return x, y

def split_dataset(x, y, train_frac=0.5, seed=0):
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)
    return (x[perm[:n_train]], y[perm[:n_train]]), (x[perm[n_train:]], y[perm[n_train:]])

# ───────────────────────── Subspace extraction (same as before) ─────────────────────────
@torch.no_grad()
def extract_mem_subspace(model, train_x, test_x, layer=LAYER, k=K_DIMS):
    model.eval()
    n = min(2048, len(train_x), len(test_x))
    tr, te = train_x[:n].to(DEVICE), test_x[:n].to(DEVICE)
    _, cache_tr = model.run_with_cache(tr)
    _, cache_te = model.run_with_cache(te)
    resid_tr = cache_tr[f"blocks.{layer}.hook_resid_post"].mean(1)
    resid_te = cache_te[f"blocks.{layer}.hook_resid_post"].mean(1)
    resid_tr = resid_tr - resid_tr.mean(0, keepdim=True)
    resid_te = resid_te - resid_te.mean(0, keepdim=True)
    contrast = resid_tr.mean(0) - resid_te.mean(0)
    _, _, V = torch.svd_lowrank(resid_tr, q=k + 4)
    basis = V[:, :k].T.clone()
    basis[0] = contrast
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis

@torch.no_grad()
def make_random_subspace(k=K_DIMS, d=D_MODEL):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T

# ───────────────────────── Fourier feature construction ─────────────────────────
def construct_fourier_targets(x, k_max=8):
    """
    Build classic modular-addition Fourier features for each example.
    Returns a matrix of shape (N, 4*k_max) containing:
      cos(2π k a / p), sin(2π k a / p),
      cos(2π k b / p), sin(2π k b / p),
      cos(2π k (a+b) / p), sin(2π k (a+b) / p)   for k = 1..k_max
    (We keep a richer set than the minimal one.)
    """
    a = x[:, 0].float()
    b = x[:, 1].float()
    s = (a + b) % P
    features = []
    for k in range(1, k_max + 1):
        for var in [a, b, s]:
            angle = 2 * np.pi * k * var / P
            features.append(torch.cos(angle))
            features.append(torch.sin(angle))
    return torch.stack(features, dim=1).to(DEVICE)   # (N, n_features)

@torch.no_grad()
def extract_fourier_subspace(model, x, layer=LAYER, k_out=K_DIMS, k_max=8):
    """
    Find residual-stream directions that best linearly read out
    the classic Fourier features (via least squares).
    """
    model.eval()
    # Use a subset for speed / stability
    n = min(4096, len(x))
    xb = x[:n].to(DEVICE)
    targets = construct_fourier_targets(xb, k_max=k_max)   # (n, F)

    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{layer}.hook_resid_post"].mean(1)  # (n, d)

    # Center
    resid = resid - resid.mean(0, keepdim=True)
    targets = targets - targets.mean(0, keepdim=True)

    # Least-squares: resid @ W ≈ targets  →  W = resid⁺ @ targets
    # We want directions in residual space, so we take the top right singular vectors
    # of the map from residual → Fourier features.
    # Solve for the best linear map and then SVD it.
    # resid: (n, d), targets: (n, F)
    # Use torch.linalg.lstsq or manual pseudo-inverse
    W = torch.linalg.lstsq(resid, targets).solution   # (d, F)

    # Now take the top principal directions of W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # U columns are directions in residual space ordered by importance
    basis = U[:, :k_out].T   # (k_out, d)
    # Orthonormalize just in case
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis

# ───────────────────────── Overlap metric ─────────────────────────
@torch.no_grad()
def subspace_overlap(A, B):
    """
    Compute the subspace overlap (sum of squared cosines of principal angles).
    A, B: (k, d) orthonormal bases.
    Returns a scalar in [0, 1]. 1 = identical subspaces, 0 = orthogonal.
    """
    # Principal angles via SVD of A @ B.T
    M = A @ B.T
    _, S, _ = torch.linalg.svd(M)
    # S contains the cosines of the principal angles
    return (S ** 2).sum().item() / min(A.shape[0], B.shape[0])

# ───────────────────────── Causal hooks ─────────────────────────
def make_phase_cancel_hook(B, coeff=2.0):
    B = B.detach()
    def hook_fn(resid, hook):
        flat = resid.reshape(-1, resid.shape[-1])
        proj = (flat @ B.T) @ B
        flat = flat - coeff * proj
        return flat.reshape(resid.shape)
    return hook_fn

@torch.no_grad()
def evaluate(model, test_x, test_y, hook=None):
    model.eval()
    if hook is not None:
        with model.hooks(fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook)]):
            logits = model(test_x.to(DEVICE))
    else:
        logits = model(test_x.to(DEVICE))
    loss = F.cross_entropy(logits[:, -1], test_y.to(DEVICE)).item()
    acc  = (logits[:, -1].argmax(-1) == test_y.to(DEVICE)).float().mean().item()
    return loss, acc

# ───────────────────────── Main test ─────────────────────────
def run_fourier_test(checkpoint_path, seed):
    print(f"\n===== Fourier Overlap + Causal Test | seed {seed} =====")
    model = make_model(seed).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed=seed)

    print("Extracting your memory subspace...")
    mem_B = extract_mem_subspace(model, train_x, test_x)

    print("Extracting Nanda-style Fourier subspace...")
    fourier_B = extract_fourier_subspace(model, train_x)

    print("Computing geometric overlap...")
    overlap = subspace_overlap(mem_B, fourier_B)
    print(f"  Subspace overlap = {overlap:.4f}  (0 = orthogonal, 1 = identical)")

    # Causal tests
    results = {
        "seed": seed,
        "overlap": overlap,
        "causal": {}
    }

    tests = {
        "mem": mem_B,
        "fourier": fourier_B,
        "union": torch.cat([mem_B, fourier_B], dim=0),
        "random": make_random_subspace(k=mem_B.shape[0] + fourier_B.shape[0]),
    }
    # Re-orthonormalize union and random
    tests["union"] = torch.linalg.qr(tests["union"].T, mode="reduced")[0].T
    tests["random"] = torch.linalg.qr(tests["random"].T, mode="reduced")[0].T

    for name, B in tests.items():
        print(f"\nCausal test: {name}")
        results["causal"][name] = {}
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_phase_cancel_hook(B, coeff=coeff)
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            results["causal"][name][str(coeff)] = {"loss": loss, "acc": acc}
            print(f"  c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}")

    # Recovery
    loss_r, acc_r = evaluate(model, test_x, test_y, hook=None)
    results["recovery"] = {"loss": loss_r, "acc": acc_r}

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"fourier_test_seed{seed}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    return results

# ───────────────────────── Entry point ─────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    run_fourier_test(args.checkpoint, args.seed)
