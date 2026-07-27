#!/usr/bin/env python3
"""
logit_contribution_mult_512.py

Measure how much of the correct logit is carried by the contrastive ("mem")
subspace vs algorithmic / random / complement subspaces
on already-trained modular multiplication models (d_model=512, 2 layers).
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
N_LAYERS = 2
N_HEADS = 8
D_HEAD = 64
D_MLP = 2048
K_DIMS = 16
LAYER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_logit_mult_512")
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
    y = ((aa * bb) % P).flatten()
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


def construct_trig_targets(x, k_max=8):
    a = x[:, 0].float()
    b = x[:, 1].float()
    s = (a * b) % P
    feats = []
    for k in range(1, k_max + 1):
        for var in (a, b, s):
            angle = 2 * torch.pi * k * var / P
            feats.append(torch.cos(angle))
            feats.append(torch.sin(angle))
    return torch.stack(feats, dim=1).to(DEVICE)


@torch.no_grad()
def extract_algorithmic_subspace(model, x, layer=LAYER, k_out=K_DIMS, k_max=8):
    model.eval()
    n = min(4096, len(x))
    xb = x[:n].to(DEVICE)
    targets = construct_trig_targets(xb, k_max=k_max)
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
def measure_logit_contribution(model, test_x, test_y, subspaces, max_samples=4096):
    model.eval()
    model.reset_hooks()
    n = min(max_samples, len(test_x))
    xb = test_x[:n].to(DEVICE)
    yb = test_y[:n].to(DEVICE)

    _, cache = model.run_with_cache(xb)
    resid = cache[f"blocks.{LAYER}.hook_resid_post"].mean(1)   # (n, d)
    W_U = model.W_U  # (d, vocab)

    full_logits = resid @ W_U
    full_correct = full_logits.gather(1, yb.unsqueeze(1)).squeeze(1)
    full_mean = full_correct.mean().item()

    results = {"full_correct_logit_mean": full_mean}

    for name, B in subspaces.items():
        # project residual onto subspace
        proj = (resid @ B.T) @ B
        logits_proj = proj @ W_U
        correct_proj = logits_proj.gather(1, yb.unsqueeze(1)).squeeze(1)
        mean_proj = correct_proj.mean().item()
        frac = mean_proj / (full_mean + 1e-8)

        # also measure after removing the subspace
        resid_removed = resid - proj
        logits_rem = resid_removed @ W_U
        correct_rem = logits_rem.gather(1, yb.unsqueeze(1)).squeeze(1)
        mean_rem = correct_rem.mean().item()
        frac_rem = mean_rem / (full_mean + 1e-8)

        results[name] = {
            "proj_correct_logit_mean": mean_proj,
            "frac_of_full": frac,
            "after_removal_logit_mean": mean_rem,
            "frac_remaining": frac_rem,
        }
        print(f"  {name:12s}  frac={frac:.3f}  remaining={frac_rem:.3f}")

    return results


def run_one(checkpoint_path, seed):
    print(f"\n===== Logit contribution (multiplication) | seed {seed} =====")
    print(f"Loading {checkpoint_path}")

    model = make_model().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    x, y = make_dataset()
    (train_x, _), (test_x, test_y) = split_dataset(x, y, seed)

    # quick sanity
    with torch.no_grad():
        logits = model(test_x[:512].to(DEVICE))[:, -1]
        acc = (logits.argmax(-1) == test_y[:512].to(DEVICE)).float().mean().item()
    print(f"  Quick acc check: {acc:.3f}")
    if acc < 0.99:
        print("  Not fully grokked — skipping")
        return None

    print("  Extracting subspaces...")
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    algo_B = extract_algorithmic_subspace(model, train_x)
    rand_B = make_random_subspace(K_DIMS, D_MODEL)

    # complement of mem
    # simple: random directions orthogonalized against mem
    Q, _ = torch.linalg.qr(torch.cat([mem_B, torch.randn(D_MODEL - K_DIMS, D_MODEL, device=DEVICE)], dim=0).T)
    complement_B = Q[:, K_DIMS:].T[:K_DIMS]

    subspaces = {
        "mem": mem_B,
        "algorithmic": algo_B,
        "random": rand_B,
        "complement": complement_B,
    }

    print("  Measuring logit contribution...")
    contrib = measure_logit_contribution(model, test_x, test_y, subspaces)

    return {
        "seed": seed,
        "task": "multiplication",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "logit_contribution": contrib,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="results_mult_512")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        candidates = [
            Path(args.ckpt_dir) / f"grokked_mult_d512_seed{seed}.pt",
            Path(args.ckpt_dir) / f"grokked_d512_seed{seed}.pt",
            Path(".") / f"grokked_mult_d512_seed{seed}.pt",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            print(f"Checkpoint not found for seed {seed} — skipping")
            continue

        r = run_one(ckpt, seed)
        if r is not None:
            all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"logit_mult_d512_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
