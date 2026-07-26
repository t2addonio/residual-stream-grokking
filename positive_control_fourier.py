#!/usr/bin/env python3
"""
positive_control_fourier.py

Test D — Positive Control
Show that the measurement pipeline can recover known Nanda-style Fourier features.
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
import json
from datetime import datetime
import argparse

P = 113
K_DIMS = 16
LAYER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_positive_control")
RESULTS_DIR.mkdir(exist_ok=True)


def make_model(d_model: int, seed: int = 0):
    torch.manual_seed(seed)
    if d_model == 256:
        n_heads, d_head, d_mlp = 4, 64, 1024
    else:
        n_heads, d_head, d_mlp = 4, 32, 512
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=d_model, n_heads=n_heads, d_head=d_head,
        d_mlp=d_mlp, act_fn="relu", normalization_type=None,
        d_vocab=P, n_ctx=2, device=DEVICE,
    )
    return HookedTransformer(cfg)


def make_dataset():
    a = torch.arange(P)
    b = torch.arange(P)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    y = ((aa + bb) % P).flatten()
    return x, y


def construct_synthetic_fourier_targets(x, k_max=8):
    a, b = x[:, 0].float(), x[:, 1].float()
    s = (a + b) % P
    feats = []
    for k in range(1, k_max + 1):
        for var in (a, b, s):
            angle = 2 * torch.pi * k * var / P
            feats.append(torch.cos(angle))
            feats.append(torch.sin(angle))
    return torch.stack(feats, dim=1).to(DEVICE)


@torch.no_grad()
def extract_fourier_subspace_from_model(model, x, k_out=K_DIMS, k_max=8, layer=LAYER):
    model.eval()
    n = min(4096, len(x))
    xb = x[:n].to(DEVICE)
    targets = construct_synthetic_fourier_targets(xb, k_max=k_max)
    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{layer}.hook_resid_post"].mean(1)
    resid = resid - resid.mean(0, keepdim=True)
    targets = targets - targets.mean(0, keepdim=True)
    W = torch.linalg.lstsq(resid, targets).solution
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    basis = U[:, :k_out].T
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis, S[:k_out].cpu().numpy()


def subspace_overlap(A, B):
    M = A @ B.T
    _, S, _ = torch.linalg.svd(M)
    return (S ** 2).sum().item() / min(A.shape[0], B.shape[0])


def run_positive_control(checkpoint_path, d_model, seed):
    print(f"\n===== Positive Control | seed {seed} | d_model={d_model} =====")
    model = make_model(d_model, seed).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    x, y = make_dataset()

    print("Extracting Fourier subspace (subset 1)...")
    fourier_1, singular_values = extract_fourier_subspace_from_model(model, x)
    print("Extracting Fourier subspace (subset 2)...")
    fourier_2, _ = extract_fourier_subspace_from_model(model, x[1000:])

    overlap_stability = subspace_overlap(fourier_1, fourier_2)
    print(f"  Stability overlap = {overlap_stability:.4f}")
    print(f"  Top singular values: {singular_values[:8]}")

    result = {
        "seed": seed, "d_model": d_model, "checkpoint": str(checkpoint_path),
        "stability_overlap": float(overlap_stability),
        "top_singular_values": singular_values.tolist(),
        "note": "High stability_overlap (>0.7) indicates the pipeline reliably recovers consistent Fourier directions."
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"positive_control_d{d_model}_seed{seed}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--d_model", type=int, required=True, choices=[128, 256])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_positive_control(args.checkpoint, args.d_model, args.seed)
