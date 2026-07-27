#!/usr/bin/env python3
"""
run_discovery_512.py

Full discovery pipeline for modular addition at d_model=512:
  train → contrastive subspace → Fourier control → overlap → causal ablation

Usage examples:
  CUDA_VISIBLE_DEVICES=0 python run_discovery_512.py --start 13000 --end 13004
  CUDA_VISIBLE_DEVICES=1 python run_discovery_512.py --start 13005 --end 13009
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_discovery_512")
RESULTS_DIR.mkdir(exist_ok=True)


def make_model(seed: int):
    torch.manual_seed(seed)
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
def evaluate(model, test_x, test_y, hook=None):
    model.eval()
    if hook is not None:
        with model.hooks(fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook)]):
            logits = model(test_x.to(DEVICE))
    else:
        logits = model(test_x.to(DEVICE))
    loss = F.cross_entropy(logits[:, -1], test_y.to(DEVICE)).item()
    acc = (logits[:, -1].argmax(-1) == test_y.to(DEVICE)).float().mean().item()
    return loss, acc


def run_one(seed, lr=1e-3, weight_decay=1.0):
    print(f"\n===== Addition | Seed {seed} | d_model={D_MODEL} =====")
    torch.manual_seed(seed)
    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed)

    model = make_model(seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(train_x), (BATCH_SIZE,))
        xb, yb = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)
        logits = model(xb)
        loss = F.cross_entropy(logits[:, -1], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == MAX_STEPS - 1:
            tl, ta = evaluate(model, test_x, test_y)
            print(f"  step {step:5d}  test_loss={tl:.3f}  acc={ta:.3f}")
            if tl < 0.5:
                print("  → grokked")
                break

    final_loss, final_acc = evaluate(model, test_x, test_y)
    grokked = final_loss < 0.8
    result = {
        "seed": seed,
        "d_model": D_MODEL,
        "task": "addition",
        "final_test_loss": final_loss,
        "final_test_acc": final_acc,
        "grokked": grokked,
        "overlap": None,
        "causal": None,
    }
    if not grokked:
        print("  Did not grok – skipping causal tests")
        return result

    ckpt = RESULTS_DIR / f"grokked_d{D_MODEL}_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved checkpoint → {ckpt}")

    print("  Extracting subspaces...")
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    fourier_B = extract_fourier_subspace(model, train_x)
    overlap = subspace_overlap(mem_B, fourier_B)
    print(f"  Overlap = {overlap:.5f}")

    union_B = torch.cat([mem_B, fourier_B], dim=0)
    union_B = torch.linalg.qr(union_B.T, mode="reduced")[0].T
    rand_B = make_random_subspace(mem_B.shape[0] + fourier_B.shape[0], D_MODEL)

    causal = {}
    for name, B in [("mem", mem_B), ("fourier", fourier_B), ("union", union_B), ("random", rand_B)]:
        causal[name] = {}
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_cancel_hook(B, coeff)
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            causal[name][str(coeff)] = {"loss": loss, "acc": acc}
            print(f"  {name:8s} c={coeff:.1f} → acc={acc:.3f}")

    loss_r, acc_r = evaluate(model, test_x, test_y)
    causal["recovery"] = {"loss": loss_r, "acc": acc_r}
    result["overlap"] = overlap
    result["causal"] = causal
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1.0)
    args = parser.parse_args()

    all_results = []
    for seed in range(args.start, args.end + 1):
        r = run_one(seed, lr=args.lr, weight_decay=args.wd)
        all_results.append(r)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"discovery_d{D_MODEL}_{args.start}_{args.end}_{stamp}.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved progress → {out}")

    print(f"\nFinished seeds {args.start}–{args.end}. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
