#!/usr/bin/env python3
"""
layer_sweep.py

Extract the contrastive subspace at every layer and measure
causal effect + Fourier overlap at each layer.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
from pathlib import Path
import json
from datetime import datetime
import argparse

P = 113
K_DIMS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_layer_sweep")
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


def split_dataset(x, y, seed, train_frac=0.5):
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)
    return (x[perm[:n_train]], y[perm[:n_train]]), (x[perm[n_train:]], y[perm[n_train:]])


@torch.no_grad()
def extract_contrastive_subspace(model, train_x, test_x, layer, k=K_DIMS):
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
def extract_fourier_subspace(model, x, layer, k_out=K_DIMS, k_max=8):
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


def subspace_overlap(A, B):
    M = A @ B.T
    _, S, _ = torch.linalg.svd(M)
    return (S ** 2).sum().item() / min(A.shape[0], B.shape[0])


def make_cancel_hook(B, coeff=2.0):
    B = B.detach()
    def hook_fn(resid, hook):
        flat = resid.reshape(-1, resid.shape[-1])
        proj = (flat @ B.T) @ B
        return (flat - coeff * proj).reshape(resid.shape)
    return hook_fn


@torch.no_grad()
def evaluate(model, test_x, test_y, layer, hook=None):
    model.eval()
    if hook is not None:
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]):
            logits = model(test_x.to(DEVICE))
    else:
        logits = model(test_x.to(DEVICE))
    loss = F.cross_entropy(logits[:, -1], test_y.to(DEVICE)).item()
    acc = (logits[:, -1].argmax(-1) == test_y.to(DEVICE)).float().mean().item()
    return loss, acc


def run_layer_sweep(checkpoint_path, d_model, seed):
    print(f"\n===== Layer Sweep | seed {seed} | d_model={d_model} =====")
    model = make_model(d_model, seed).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed)

    n_layers = model.cfg.n_layers
    results = {"seed": seed, "d_model": d_model, "checkpoint": str(checkpoint_path), "layers": {}}

    for layer in range(n_layers):
        print(f"\n--- Layer {layer} ---")
        mem_B = extract_contrastive_subspace(model, train_x, test_x, layer)
        fourier_B = extract_fourier_subspace(model, train_x, layer)
        rand_B = make_random_subspace(K_DIMS, d_model)
        overlap = subspace_overlap(mem_B, fourier_B)
        print(f"  Overlap with Fourier = {overlap:.4f}")

        layer_result = {"overlap": overlap, "causal": {}}
        for name, B in [("mem", mem_B), ("fourier", fourier_B), ("random", rand_B)]:
            layer_result["causal"][name] = {}
            for coeff in [0.0, 1.0, 2.0]:
                hook = make_cancel_hook(B, coeff)
                loss, acc = evaluate(model, test_x, test_y, layer, hook=hook)
                layer_result["causal"][name][str(coeff)] = {"loss": loss, "acc": acc}
                print(f"  {name:8s} c={coeff:.1f} → acc={acc:.3f}")

        loss_r, acc_r = evaluate(model, test_x, test_y, layer, hook=None)
        layer_result["recovery"] = {"loss": loss_r, "acc": acc_r}
        results["layers"][str(layer)] = layer_result

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"layer_sweep_d{d_model}_seed{seed}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--d_model", type=int, required=True, choices=[128, 256])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_layer_sweep(args.checkpoint, args.d_model, args.seed)
