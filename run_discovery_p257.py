#!/usr/bin/env python3
"""
run_discovery_p257.py

Modular addition with larger modulus (p=257) at d_model=512, 2 layers.
Tests whether the contrastive residual component still appears, remains
nearly orthogonal, and is causally necessary when the task is harder.

Supports 15–30 seeds, parallel across multiple GPUs.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

P = 257                    # larger modulus
D_MODEL = 512
N_LAYERS = 2
N_HEADS = 8
D_HEAD = 64
D_MLP = 2048
K_DIMS = 16
LAYER = 1
MAX_STEPS = 40001          # may need a bit more than p=113
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_p257")
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


def construct_fourier_targets(x, k_max=16):
    """Fourier features for addition (a, b, a+b)."""
    a = x[:, 0].float()
    b = x[:, 1].float()
    s = (a + b) % P
    feats = []
    for k in range(1, k_max + 1):
        for var in (a, b, s):
            angle = 2 * torch.pi * k * var / P
            feats.append(torch.cos(angle))
            feats.append(torch.sin(angle))
    return torch.stack(feats, dim=1).to(DEVICE)


@torch.no_grad()
def extract_fourier_subspace(model, x, layer=LAYER, k_out=K_DIMS, k_max=16):
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
    model.eval()
    model.reset_hooks()
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    handle = None
    if hook is not None:
        handle = model.blocks[LAYER].hook_resid_post.add_hook(hook)
    try:
        logits = model(x)[:, -1]
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    finally:
        model.reset_hooks()
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
    return loss, acc


def run_one(seed, lr=LR, weight_decay=WEIGHT_DECAY):
    print(f"\n===== Seed {seed} | Modular Addition p={P} | d=512 L=2 =====")
    torch.manual_seed(seed)

    model = make_model().to(DEVICE)
    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    final_loss, final_acc = None, None
    grokked = False

    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(train_x), (BATCH_SIZE,))
        xb, yb = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)
        logits = model(xb)[:, -1]
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == MAX_STEPS - 1:
            tloss, tacc = evaluate(model, test_x, test_y)
            print(f"  step {step:5d}  test_loss={tloss:.4f}  acc={tacc:.3f}")
            final_loss, final_acc = tloss, tacc
            if tloss < 0.5:
                grokked = True
                print("  → grokked, stopping early")
                break

    result = {
        "seed": seed,
        "task": "addition",
        "p": P,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "final_test_loss": final_loss,
        "final_test_acc": final_acc,
        "grokked": grokked,
    }

    if not grokked:
        print("  Did not grok — skipping subspace & causal tests")
        return result

    # Save checkpoint
    ckpt_path = RESULTS_DIR / f"grokked_p257_d512_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint → {ckpt_path}")

    # Subspaces
    print("  Extracting subspaces...")
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    fourier_B = extract_fourier_subspace(model, train_x)
    rand_B = make_random_subspace(K_DIMS, D_MODEL)

    union = torch.cat([mem_B, fourier_B], dim=0)
    union = torch.linalg.qr(union.T, mode="reduced")[0].T

    overlap = subspace_overlap(mem_B, fourier_B)
    print(f"  Overlap (contrastive vs Fourier) = {overlap:.5f}")
    result["overlap"] = overlap

    # Causal ablations
    causal = {}
    for name, B in [("mem", mem_B), ("fourier", fourier_B), ("random", rand_B), ("union", union)]:
        causal[name] = {}
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_cancel_hook(B, coeff) if coeff > 0 else None
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            causal[name][f"c={coeff}"] = {"loss": loss, "acc": acc}
            print(f"  {name:10s} c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}")
    result["causal"] = causal

    # Recovery
    loss_c, acc_c = evaluate(model, test_x, test_y, hook=make_cancel_hook(mem_B, 2.0))
    loss_r, acc_r = evaluate(model, test_x, test_y)
    result["recovery"] = {"cancelled_acc": acc_c, "recovered_acc": acc_r}
    print(f"  recovery: cancelled={acc_c:.3f} recovered={acc_r:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    args = parser.parse_args()

    all_results = []
    for seed in range(args.start, args.end + 1):
        r = run_one(seed, lr=args.lr, weight_decay=args.wd)
        all_results.append(r)

        # incremental save
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"p257_d512_{args.start}_{args.end}_{stamp}.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nFinished seeds {args.start}–{args.end}")
    print(f"Results under {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
