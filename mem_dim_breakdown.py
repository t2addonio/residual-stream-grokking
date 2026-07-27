#!/usr/bin/env python3
"""
mem_dim_breakdown.py

Breaks down how much of the correct logit is carried by each of the
16 dimensions of the contrastive ("mem") subspace.
Compares seed 14025 (outlier) against normal seeds.
"""

import torch
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

SEEDS = [14025, 14000, 14015, 14020]
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
def analyze_dimensions(seed):
    ckpt = CKPT_DIR / f"grokked_d{D_MODEL}_L{N_LAYERS}_seed{seed}.pt"
    if not ckpt.exists():
        print(f"Checkpoint missing: {ckpt}")
        return

    print(f"\n===== Seed {seed} — dimension-wise logit contribution =====")
    model = make_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    x, y = make_dataset()
    (train_x, _), (test_x, test_y) = split_dataset(x, y, seed)

    mem_B = extract_contrastive_subspace(model, train_x, test_x)  # (K, d_model)

    n = min(2048, len(test_x))
    xb = test_x[:n].to(DEVICE)
    yb = test_y[:n].to(DEVICE)

    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{LAYER}.hook_resid_post"][:, -1]  # (batch, d)
    W_U = model.W_U

    # Full correct logit
    full_logits = (resid @ W_U).gather(1, yb.unsqueeze(1)).squeeze(1)
    full_mean = full_logits.mean().item()

    # Contribution of each individual dimension
    # Projection onto single direction i: (resid @ v_i) * v_i
    contribs = []
    for i in range(K_DIMS):
        v = mem_B[i]                          # (d,)
        coeff = resid @ v                     # (batch,)
        proj = coeff.unsqueeze(1) * v.unsqueeze(0)  # (batch, d)
        logit_i = (proj @ W_U).gather(1, yb.unsqueeze(1)).squeeze(1)
        mean_i = logit_i.mean().item()
        frac_i = mean_i / (full_mean + 1e-8)
        contribs.append((i, mean_i, frac_i))

    # Sort by absolute contribution
    contribs_sorted = sorted(contribs, key=lambda t: abs(t[1]), reverse=True)

    print(f"Full correct logit mean: {full_mean:.3f}")
    print(f"{'dim':>4}  {'logit':>10}  {'frac':>8}")
    print("-" * 28)
    cumulative = 0.0
    for i, mean_i, frac_i in contribs_sorted:
        cumulative += frac_i
        print(f"{i:4d}  {mean_i:10.3f}  {frac_i:8.3f}   (cumul {cumulative:.3f})")

    # Also report how many dims are needed to reach 90% and 99%
    cumul = 0.0
    needed_90 = needed_99 = K_DIMS
    for rank, (i, mean_i, frac_i) in enumerate(contribs_sorted, 1):
        cumul += frac_i
        if cumul >= 0.90 and needed_90 == K_DIMS:
            needed_90 = rank
        if cumul >= 0.99 and needed_99 == K_DIMS:
            needed_99 = rank
            break
    print(f"\nDims needed for ≥90% of correct logit: {needed_90}")
    print(f"Dims needed for ≥99% of correct logit: {needed_99}")


if __name__ == "__main__":
    for s in SEEDS:
        analyze_dimensions(s)
