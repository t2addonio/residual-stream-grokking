#!/usr/bin/env python3
"""
diagnose_14025.py
Compares seed 14025 (extreme logit contribution) against normal seeds.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig

P = 113
D_MODEL = 512
N_LAYERS = 4
N_HEADS = 8
D_HEAD = 64
D_MLP = 2048
K_DIMS = 16
LAYER = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [14025, 14000, 14015, 14020]   # outlier first, then normals
CKPT_DIR = Path("results_discovery_512_L4")


def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_head=D_HEAD, d_mlp=D_MLP, act_fn="relu",
        normalization_type=None, d_vocab=P, n_ctx=2, device=DEVICE,
    )
    return HookedTransformer(cfg)


def make_dataset():
    a = torch.arange(P)
    b = torch.arange(P)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    y = ((aa + bb) % P).flatten()
    return x, y


def split_dataset(x, y, seed, train_frac=0.5):
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)
    return (x[perm[:n_train]], y[perm[:n_train]]), (x[perm[n_train:]], y[perm[n_train:]])


@torch.no_grad()
def extract_contrastive_subspace(model, train_x, test_x, layer=LAYER, k=K_DIMS):
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
def extract_fourier_subspace(model, x, layer=LAYER, k_out=K_DIMS, k_max=8):
    model.eval()
    n = min(4096, len(x))
    xb = x[:n].to(DEVICE)
    a, b = xb[:, 0].float(), xb[:, 1].float()
    s = (a + b) % P
    feats = []
    for k in range(1, k_max + 1):
        for var in (a, b, s):
            angle = 2 * torch.pi * k * var / P
            feats.append(torch.cos(angle))
            feats.append(torch.sin(angle))
    targets = torch.stack(feats, dim=1)
    targets = targets - targets.mean(0, keepdim=True)
    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{layer}.hook_resid_post"].mean(1)
    resid = resid - resid.mean(0, keepdim=True)
    W = torch.linalg.lstsq(resid, targets).solution
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    basis = U[:, :k_out].T
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis


@torch.no_grad()
def analyze_seed(seed):
    ckpt = CKPT_DIR / f"grokked_d{D_MODEL}_L{N_LAYERS}_seed{seed}.pt"
    if not ckpt.exists():
        print(f"  Checkpoint missing: {ckpt}")
        return

    print(f"\n===== Seed {seed} =====")
    model = make_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    x, y = make_dataset()
    (train_x, _), (test_x, test_y) = split_dataset(x, y, seed)

    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    fourier_B = extract_fourier_subspace(model, train_x)

    # Residual energy
    n = min(2048, len(test_x))
    xb = test_x[:n].to(DEVICE)
    yb = test_y[:n].to(DEVICE)
    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{LAYER}.hook_resid_post"][:, -1]  # (batch, d)

    resid_norm = resid.norm(dim=-1).mean().item()
    mem_proj = (resid @ mem_B.T) @ mem_B
    mem_energy = (mem_proj.norm(dim=-1) ** 2).mean().item()
    fourier_proj = (resid @ fourier_B.T) @ fourier_B
    fourier_energy = (fourier_proj.norm(dim=-1) ** 2).mean().item()
    total_energy = (resid.norm(dim=-1) ** 2).mean().item()

    # Correct logit contribution (quick)
    W_U = model.W_U
    full_logit = (resid @ W_U).gather(1, yb.unsqueeze(1)).squeeze(1).mean().item()
    mem_logit = (mem_proj @ W_U).gather(1, yb.unsqueeze(1)).squeeze(1).mean().item()
    fourier_logit = (fourier_proj @ W_U).gather(1, yb.unsqueeze(1)).squeeze(1).mean().item()

    # Alignment of first mem direction with unembedding
    mem_dir = mem_B[0]  # primary contrastive direction
    mem_dir = mem_dir / mem_dir.norm()
    # How aligned is it with the average correct-class unembedding?
    correct_U = W_U[:, yb].mean(1)
    correct_U = correct_U / correct_U.norm()
    alignment = (mem_dir @ correct_U).item()

    print(f"  Residual norm (mean)          : {resid_norm:.3f}")
    print(f"  Fraction of residual energy in mem     : {mem_energy / total_energy:.4f}")
    print(f"  Fraction of residual energy in Fourier : {fourier_energy / total_energy:.4f}")
    print(f"  Correct logit (full)          : {full_logit:.3f}")
    print(f"  Correct logit from mem        : {mem_logit:.3f}  ({mem_logit/full_logit:.3f})")
    print(f"  Correct logit from Fourier    : {fourier_logit:.3f}")
    print(f"  Alignment mem_dir ↔ avg correct U : {alignment:.4f}")


if __name__ == "__main__":
    for s in SEEDS:
        analyze_seed(s)
