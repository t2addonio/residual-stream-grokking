#!/usr/bin/env python3
"""
logit_contribution_L4.py

Measures how much of the correct logit is carried by:
  - the contrastive ("mem"/scaffolding) subspace
  - the Fourier subspace
  - their union
  - the orthogonal complement
  - a random control subspace

Run on already-grokked 4-layer d_model=512 checkpoints.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
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
RESULTS_DIR = Path("results_logit_L4")
RESULTS_DIR.mkdir(exist_ok=True)


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
        n_ctx=2,
        device=DEVICE,
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


def construct_fourier_targets(x, k_max=8):
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
def extract_fourier_subspace(model, x, layer=LAYER, k_out=K_DIMS, k_max=8):
    model.eval()
    n = min(4096, len(x))
    xb = x[:n].to(DEVICE)
    targets = construct_fourier_targets(xb, k_max=k_max)
    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{layer}.hook_resid_post"].mean(1)
    resid = resid - resid.mean(0, keepdim=True)
    targets = targets - targets.mean(0, keepdim=True)
    W = torch.linalg.lstsq(resid, targets).solution
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    basis = U[:, :k_out].T
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis


@torch.no_grad()
def make_random_subspace(k, d):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T


@torch.no_grad()
def project_resid(resid, B):
    """Project residual onto subspace B (rows are basis vectors)."""
    return (resid @ B.T) @ B


@torch.no_grad()
def measure_logit_contribution(model, test_x, test_y, subspaces, n_samples=2048):
    """
    For each example, measure the contribution of each subspace projection
    to the correct-class logit.
    """
    model.eval()
    n = min(n_samples, len(test_x))
    xb = test_x[:n].to(DEVICE)
    yb = test_y[:n].to(DEVICE)

    # Get residual stream and unembedding
    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{LAYER}.hook_resid_post"][:, -1]  # (batch, d_model)
    W_U = model.W_U  # (d_model, d_vocab)

    # Full correct logit
    full_logits = resid @ W_U
    correct_logit_full = full_logits.gather(1, yb.unsqueeze(1)).squeeze(1)

    results = {"full_correct_logit_mean": correct_logit_full.mean().item()}

    for name, B in subspaces.items():
        if B is None:
            continue
        proj = project_resid(resid, B)
        proj_logits = proj @ W_U
        correct_logit_proj = proj_logits.gather(1, yb.unsqueeze(1)).squeeze(1)

        # Also measure the residual after removing the subspace
        resid_minus = resid - proj
        minus_logits = resid_minus @ W_U
        correct_logit_minus = minus_logits.gather(1, yb.unsqueeze(1)).squeeze(1)

        results[name] = {
            "proj_correct_logit_mean": correct_logit_proj.mean().item(),
            "frac_of_full": (correct_logit_proj / (correct_logit_full + 1e-8)).mean().item(),
            "after_removal_correct_logit_mean": correct_logit_minus.mean().item(),
            "frac_remaining_after_removal": (correct_logit_minus / (correct_logit_full + 1e-8)).mean().item(),
        }

    return results


def run_one(checkpoint_path, seed):
    print(f"\n===== Logit contribution | seed {seed} | L=4 d=512 =====")
    print(f"Loading {checkpoint_path}")

    model = make_model().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed)

    print("Extracting subspaces...")
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    fourier_B = extract_fourier_subspace(model, train_x)

    # Orthogonal complement of the contrastive subspace
    # (project onto nullspace of mem_B)
    eye = torch.eye(D_MODEL, device=DEVICE)
    P_mem = mem_B.T @ mem_B
    complement_basis = eye - P_mem
    # orthonormalize the complement (take top singular vectors)
    U, S, _ = torch.linalg.svd(complement_basis, full_matrices=False)
    # keep the directions with large singular values
    keep = (S > 0.5).sum().item()
    complement_B = U[:, :keep].T  # may be large; we only need it for projection

    rand_B = make_random_subspace(K_DIMS, D_MODEL)

    subspaces = {
        "mem": mem_B,
        "fourier": fourier_B,
        "random": rand_B,
        "complement_of_mem": complement_B,
    }

    print("Measuring logit contributions...")
    contrib = measure_logit_contribution(model, test_x, test_y, subspaces)

    # Pretty print
    print(f"  Full correct logit (mean)          = {contrib['full_correct_logit_mean']:.3f}")
    for name in ["mem", "fourier", "random", "complement_of_mem"]:
        if name not in contrib:
            continue
        r = contrib[name]
        print(f"  {name:20s}  frac={r['frac_of_full']:.3f}  "
              f"after_removal_frac={r['frac_remaining_after_removal']:.3f}")

    result = {
        "seed": seed,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "layer": LAYER,
        "checkpoint": str(checkpoint_path),
        "logit_contribution": contrib,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds whose checkpoints to analyze")
    parser.add_argument("--ckpt_dir", type=str, default="results_discovery_512_L4")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        ckpt = Path(args.ckpt_dir) / f"grokked_d{D_MODEL}_L{N_LAYERS}_seed{seed}.pt"
        if not ckpt.exists():
            print(f"Checkpoint not found: {ckpt} — skipping")
            continue
        r = run_one(ckpt, seed)
        all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"logit_L4_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
