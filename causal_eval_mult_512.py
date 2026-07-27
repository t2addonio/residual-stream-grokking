#!/usr/bin/env python3
"""
causal_eval_mult_512.py

Clean causal evaluation on already-trained modular multiplication
checkpoints (d_model=512, 2 layers).

Fixes the previous bug where hooks leaked between tests.
Every ablation starts from a fresh forward pass with no residual hooks.
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
RESULTS_DIR = Path("results_causal_mult_512")
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
    y = ((aa * bb) % P).flatten()          # MULTIPLICATION
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
    tr = train_x[:n].to(DEVICE)
    te = test_x[:n].to(DEVICE)
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


def subspace_overlap(A, B):
    M = A @ B.T
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    return (S ** 2).sum().item() / min(A.shape[0], B.shape[0])


@torch.no_grad()
def make_random_subspace(k, d):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T


def make_cancel_hook(B, coeff=2.0):
    def hook(resid, hook):
        proj = (resid @ B.T) @ B
        return resid - coeff * proj
    return hook


@torch.no_grad()
def evaluate(model, x, y, hook=None):
    """Hardened evaluation — always starts from a completely clean model."""
    model.eval()
    model.reset_hooks()          # ← critical: wipe every residual hook

    x = x.to(DEVICE)
    y = y.to(DEVICE)

    if hook is not None:
        handle = model.blocks[LAYER].hook_resid_post.add_hook(hook)
    else:
        handle = None

    try:
        logits = model(x)[:, -1]
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    finally:
        model.reset_hooks()      # ← clear again after the forward pass
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    return loss, acc


def run_one(checkpoint_path, seed):
    print(f"\n===== Causal eval (multiplication) | seed {seed} | d=512 L=2 =====")
    print(f"Loading {checkpoint_path}")

    model = make_model().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed)

    # Baseline (no intervention)
    base_loss, base_acc = evaluate(model, test_x, test_y)
    print(f"  Baseline: loss={base_loss:.4f} acc={base_acc:.3f}")

    if base_acc < 0.99:
        print("  Model has not grokked — skipping")
        return {
            "seed": seed,
            "task": "multiplication",
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "final_test_loss": base_loss,
            "final_test_acc": base_acc,
            "grokked": False,
        }

    print("  Extracting subspaces...")
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    algo_B = extract_algorithmic_subspace(model, train_x)
    rand_B = make_random_subspace(K_DIMS, D_MODEL)

    # Union
    union = torch.cat([mem_B, algo_B], dim=0)
    union = torch.linalg.qr(union.T, mode="reduced")[0].T

    overlap = subspace_overlap(mem_B, algo_B)
    print(f"  Overlap (contrastive vs algorithmic) = {overlap:.5f}")

    causal = {}
    for name, B in [("mem", mem_B), ("algorithmic", algo_B), ("random", rand_B), ("union", union)]:
        causal[name] = {}
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_cancel_hook(B, coeff) if coeff > 0 else None
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            causal[name][f"c={coeff}"] = {"loss": loss, "acc": acc}
            print(f"  {name:12s} c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}")

    # Explicit recovery test
    print("  Recovery test...")
    loss_c, acc_c = evaluate(model, test_x, test_y, hook=make_cancel_hook(mem_B, 2.0))
    loss_r, acc_r = evaluate(model, test_x, test_y)  # clean
    print(f"  cancelled acc={acc_c:.3f}  recovered acc={acc_r:.3f}")

    return {
        "seed": seed,
        "task": "multiplication",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "final_test_loss": base_loss,
        "final_test_acc": base_acc,
        "grokked": True,
        "overlap": overlap,
        "causal": causal,
        "recovery": {"cancelled_acc": acc_c, "recovered_acc": acc_r},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="results_mult_512")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        # try common naming patterns
        candidates = [
            Path(args.ckpt_dir) / f"grokked_mult_d512_seed{seed}.pt",
            Path(args.ckpt_dir) / f"grokked_d512_seed{seed}.pt",
            Path("results_mult_512") / f"grokked_mult_d512_seed{seed}.pt",
            Path(".") / f"grokked_mult_d512_seed{seed}.pt",
        ]
        ckpt = None
        for c in candidates:
            if c.exists():
                ckpt = c
                break
        if ckpt is None:
            print(f"Checkpoint not found for seed {seed} — skipping")
            continue

        r = run_one(ckpt, seed)
        all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"causal_mult_d512_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
