#!/usr/bin/env python3
"""
group_ablation.py
Cancel top-k dimensions of the contrastive subspace together.
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

N_BITS = 20
K_SPARSE = 4
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
K_DIMS = 16
LAYER = 1
N_TEST = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results_group_ablation")
OUT_DIR.mkdir(exist_ok=True)


def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_head=D_HEAD, d_mlp=D_MLP, act_fn="relu",
        normalization_type=None, d_vocab=2, n_ctx=N_BITS, device=DEVICE,
    )
    return HookedTransformer(cfg)


def make_dataset(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, N_BITS), generator=g)
    y = x[:, :K_SPARSE].sum(dim=1) % 2
    return x, y


def get_residuals(model, x):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        return cache[f"blocks.{LAYER}.hook_resid_post"].detach()


def extract_contrastive_subspace(model, train_x, test_x, k=K_DIMS):
    r_tr = get_residuals(model, train_x).mean(dim=1)
    r_te = get_residuals(model, test_x).mean(dim=1)
    mu_tr, mu_te = r_tr.mean(0), r_te.mean(0)
    contrast = mu_tr - mu_te
    centered = r_tr - mu_tr
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    B = Vh[:k].clone()
    B[0] = contrast / (contrast.norm() + 1e-8)
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B


def make_cancel_hook(B_sub, coeff=2.0):
    def hook(act, hook):
        flat = act.reshape(-1, act.shape[-1])
        coeffs = flat @ B_sub.T
        flat = flat - coeff * (coeffs @ B_sub)
        return flat.reshape(act.shape)
    return hook


def evaluate(model, x, y, hook=None):
    model.eval()
    with torch.no_grad():
        h = None
        if hook is not None:
            h = model.blocks[LAYER].hook_resid_post.add_hook(hook)
        logits = model(x.to(DEVICE))[:, -1, :]
        if h is not None:
            h.remove()
        loss = F.cross_entropy(logits, y.to(DEVICE)).item()
        acc = (logits.argmax(-1) == y.to(DEVICE)).float().mean().item()
    return loss, acc


def rank_dims_by_energy(model, x, B):
    resid = get_residuals(model, x)[:, -1, :]
    coeffs = resid @ B.T
    energy = coeffs.abs().mean(dim=0)
    order = torch.argsort(energy, descending=True).tolist()
    return order, energy.tolist()


def run_one(seed, ckpt_path, ks=(1, 2, 4, 8, 16)):
    print(f"\n===== GROUP ABLATION seed {seed} =====", flush=True)
    result = {"seed": seed, "checkpoint": str(ckpt_path), "task": "parity"}

    model = make_model().to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()

    train_x, _ = make_dataset(8000, seed)
    test_x, test_y = make_dataset(N_TEST, seed + 10000)

    B = extract_contrastive_subspace(model, train_x, test_x)
    order, energy = rank_dims_by_energy(model, test_x, B)
    result["dim_order_by_energy"] = order
    result["dim_energy"] = energy
    print(f"  energy rank: {order[:8]} ...", flush=True)

    loss0, acc0 = evaluate(model, test_x, test_y)
    result["baseline"] = {"loss": loss0, "acc": acc0}
    print(f"  baseline  loss={loss0:.4f} acc={acc0:.3f}", flush=True)

    loss_f, acc_f = evaluate(model, test_x, test_y, hook=make_cancel_hook(B, 2.0))
    result["full_cancel"] = {"loss": loss_f, "acc": acc_f}
    print(f"  full 16   loss={loss_f:.4f} acc={acc_f:.3f}", flush=True)

    group = {}
    for k in ks:
        dims = order[:k]
        B_sub = B[dims]
        loss, acc = evaluate(model, test_x, test_y, hook=make_cancel_hook(B_sub, 2.0))
        group[f"top{k}"] = {"dims": dims, "loss": loss, "acc": acc}
        print(f"  top-{k:<2}    loss={loss:.4f} acc={acc:.3f}  dims={dims}", flush=True)
    result["group_by_energy"] = group

    random.seed(seed)
    rand_group = {}
    for k in ks:
        if k >= K_DIMS:
            continue
        trials = []
        for t in range(3):
            dims = random.sample(range(K_DIMS), k)
            B_sub = B[dims]
            loss, acc = evaluate(model, test_x, test_y, hook=make_cancel_hook(B_sub, 2.0))
            trials.append({"dims": dims, "loss": loss, "acc": acc})
        rand_group[f"rand{k}"] = trials
        mean_acc = sum(t["acc"] for t in trials) / len(trials)
        print(f"  rand-{k:<2}   mean_acc={mean_acc:.3f}", flush=True)
    result["group_random"] = rand_group
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default="results_sparse_parity")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    args = p.parse_args()

    all_results = []
    for seed in args.seeds:
        candidates = [
            Path(args.ckpt_dir) / f"grokked_parity_seed{seed}.pt",
            Path(f"grokked_parity_seed{seed}.pt"),
        ]
        ckpt = None
        for c in candidates:
            if c.exists():
                ckpt = c
                break
        if ckpt is None:
            print(f"Checkpoint not found for seed {seed} — skip", flush=True)
            continue
        r = run_one(seed, ckpt)
        all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"group_ablation_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()

