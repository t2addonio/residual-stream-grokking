#!/usr/bin/env python3
"""
subspace_orthogonality_over_time.py

Train a few seeds, save the contrastive subspace at multiple steps,
then compute pairwise principal-angle overlaps between those subspaces.

Usage:
  CUDA_VISIBLE_DEVICES=0 python subspace_orthogonality_over_time.py --start 16000 --end 16004
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
MAX_STEPS = 25001
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results_ortho_time")
OUT_DIR.mkdir(exist_ok=True)

EXTRACT_STEPS = [2000, 5000, 10000, 15000, 20000]


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


def get_residuals(model, x):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        return cache[f"blocks.{LAYER}.hook_resid_post"].detach()


def extract_contrastive(model, train_x, test_x, k=K_DIMS):
    r_tr = get_residuals(model, train_x)
    r_te = get_residuals(model, test_x)
    mu_tr = r_tr.mean(dim=(0, 1))
    mu_te = r_te.mean(dim=(0, 1))
    diff = mu_tr - mu_te
    flat = r_tr.reshape(-1, D_MODEL) - mu_tr
    flat = torch.cat([diff.unsqueeze(0), flat], dim=0)
    _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
    B = Vh[:k]
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B.cpu()


def subspace_overlap(A, B):
    """Sum of squared cosines of principal angles / min(k)."""
    A = A.to(DEVICE).float()
    B = B.to(DEVICE).float()
    M = A @ B.T
    s = torch.linalg.svdvals(M)
    return (s**2).sum().item() / min(A.shape[0], B.shape[0])


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        loss = F.cross_entropy(logits[:, -1], y.to(DEVICE))
        acc = (logits[:, -1].argmax(-1) == y.to(DEVICE)).float().mean()
    return loss.item(), acc.item()


def run_one(seed):
    print(f"\n===== Seed {seed} =====")
    torch.manual_seed(seed)
    model = make_model().to(DEVICE)
    (tr_x, tr_y), (te_x, te_y) = split_dataset(*make_dataset(), seed=seed)
    tr_x, tr_y = tr_x.to(DEVICE), tr_y.to(DEVICE)
    te_x, te_y = te_x.to(DEVICE), te_y.to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    subspaces = {}
    timeline = []

    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(tr_x), (BATCH_SIZE,), device=DEVICE)
        loss = F.cross_entropy(model(tr_x[idx])[:, -1], tr_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

        if step in EXTRACT_STEPS or step == MAX_STEPS - 1:
            loss_t, acc_t = evaluate(model, te_x, te_y)
            print(f"  step {step:5d}  loss={loss_t:.4f}  acc={acc_t:.3f}")
            B = extract_contrastive(model, tr_x, te_x)
            subspaces[step] = B
            # save the basis itself
            torch.save(B, OUT_DIR / f"B_seed{seed}_step{step}.pt")
            timeline.append({"step": step, "loss": loss_t, "acc": acc_t})
            if acc_t > 0.99 and step >= 10000:
                print("  → grokked, stopping")
                break

    # Pairwise overlap matrix
    steps = sorted(subspaces.keys())
    overlap = {}
    print("  Pairwise overlaps:")
    for i, s1 in enumerate(steps):
        for s2 in steps[i:]:
            o = subspace_overlap(subspaces[s1], subspaces[s2])
            overlap[f"{s1}_vs_{s2}"] = o
            print(f"    {s1:5d} vs {s2:5d}  →  {o:.4f}")

    result = {
        "seed": seed,
        "timeline": timeline,
        "overlap": overlap,
        "steps": steps,
        "final_acc": timeline[-1]["acc"] if timeline else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    all_results = []
    for seed in range(args.start, args.end + 1):
        r = run_one(seed)
        all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"ortho_time_{args.start}_{args.end}_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
