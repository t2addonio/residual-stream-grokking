#!/usr/bin/env python3
"""
orthogonal_complement_sweep.py

Test whether additional causally-important "memory" directions
exist in the orthogonal complement of the original subspace.
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
RESULTS_DIR = Path("results_complement")
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

# ───────────────────────── Subspace helpers ─────────────────────────
@torch.no_grad()
def extract_contrastive_subspace(model, train_x, test_x, layer=LAYER, k=K_DIMS, existing_B=None):
    """
    Extract top contrastive directions.
    If existing_B is provided, first project residual stream into its orthogonal complement.
    """
    model.eval()
    n = min(2048, len(train_x), len(test_x))
    tr = train_x[:n].to(DEVICE)
    te = test_x[:n].to(DEVICE)

    _, cache_tr = model.run_with_cache(tr)
    _, cache_te = model.run_with_cache(te)

    resid_tr = cache_tr[f"blocks.{layer}.hook_resid_post"].mean(1)  # (n, d)
    resid_te = cache_te[f"blocks.{layer}.hook_resid_post"].mean(1)

    # Project into orthogonal complement of existing_B if given
    if existing_B is not None:
        # existing_B: (k, d)
        # Project out the span of existing_B
        resid_tr = resid_tr - (resid_tr @ existing_B.T) @ existing_B
        resid_te = resid_te - (resid_te @ existing_B.T) @ existing_B

    # Center
    resid_tr = resid_tr - resid_tr.mean(0, keepdim=True)
    resid_te = resid_te - resid_te.mean(0, keepdim=True)

    # Contrast direction
    contrast = resid_tr.mean(0) - resid_te.mean(0)

    # SVD on train residuals for stable basis
    _, _, V = torch.svd_lowrank(resid_tr, q=k + 4)
    basis = V[:, :k].T.clone()          # (k, d)
    basis[0] = contrast                 # force first direction to be the contrast
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis

@torch.no_grad()
def make_random_subspace(k=K_DIMS, d=D_MODEL):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T

# ───────────────────────── Hooks & Eval ─────────────────────────
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

# ───────────────────────── Main analysis ─────────────────────────
def run_complement_sweep(checkpoint_path, seed, max_rounds=3):
    print(f"\n===== Orthogonal Complement Sweep | seed {seed} =====")
    print(f"Loading {checkpoint_path}")

    model = make_model(seed).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed=seed)

    results = {
        "seed": seed,
        "checkpoint": checkpoint_path,
        "rounds": []
    }

    current_B = None          # will accumulate the union of all extracted subspaces
    accumulated_basis = []

    for round_idx in range(max_rounds):
        print(f"\n--- Round {round_idx + 1} ---")

        # Extract new subspace in the current orthogonal complement
        new_B = extract_contrastive_subspace(
            model, train_x, test_x,
            k=K_DIMS,
            existing_B=current_B
        )

        # Test causal importance of this new subspace
        print("Testing causal effect of new subspace...")
        round_result = {"round": round_idx + 1, "mem": {}, "random": {}}

        # Memory cancellation
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_phase_cancel_hook(new_B, coeff=coeff)
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            round_result["mem"][str(coeff)] = {"loss": loss, "acc": acc}
            print(f"  mem   c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}")

        # Matched random control
        rand_B = make_random_subspace()
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_phase_cancel_hook(rand_B, coeff=coeff)
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            round_result["random"][str(coeff)] = {"loss": loss, "acc": acc}
            print(f"  rand  c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}")

        # Recovery
        loss_r, acc_r = evaluate(model, test_x, test_y, hook=None)
        round_result["recovery"] = {"loss": loss_r, "acc": acc_r}
        print(f"  recovery → loss={loss_r:.4f} acc={acc_r:.3f}")

        results["rounds"].append(round_result)

        # Accumulate basis for next round (project further into complement)
        accumulated_basis.append(new_B)
        current_B = torch.cat(accumulated_basis, dim=0)
        # Re-orthonormalize
        current_B = torch.linalg.qr(current_B.T, mode="reduced")[0].T

        # Early stop if the new subspace is no longer causal
        if round_result["mem"]["2.0"]["acc"] > 0.5:
            print("New subspace is no longer strongly causal. Stopping.")
            break

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"complement_seed{seed}_{stamp}.json"
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
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    run_complement_sweep(args.checkpoint, args.seed, max_rounds=args.rounds)
