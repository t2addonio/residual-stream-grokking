#!/usr/bin/env python3
"""
run_phase_256.py

Full causal experiment at d_model=256:
- Train until grokking
- Extract memorization subspace
- Phase-cancel it (and a matched random subspace)
- Measure the collapse

Usage:
  python run_phase_256.py --seeds 8000 8001 8002 8003 8004
  python run_phase_256.py --seed 8000 --alpha 0.35
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ───────────────────────── Scaled Config (d_model=256) ─────────────────────────
P = 113
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64          # 256 // 4
D_MLP = 1024         # 4 × d_model
K_DIMS = 16          # slightly larger subspace for the wider model
LAYER = 1
MAX_STEPS = 25001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────────── Data ─────────────────────────
def make_dataset(p=P):
    a = torch.arange(p)
    b = torch.arange(p)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([aa.flatten(), bb.flatten()], dim=1)
    y = ((aa + bb) % p).flatten()
    return x, y

def split_dataset(x, y, train_frac=0.5, seed=0):
    n = len(x)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)
    return (x[perm[:n_train]], y[perm[:n_train]]), (x[perm[n_train:]], y[perm[n_train:]])

# ───────────────────────── Model ─────────────────────────
def make_model(seed=0):
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

# ───────────────────────── Subspace extraction ─────────────────────────
@torch.no_grad()
def extract_mem_subspace(model, train_x, test_x, layer=LAYER, k=K_DIMS):
    model.eval()
    n = min(2048, len(train_x), len(test_x))
    tr = train_x[:n].to(DEVICE)
    te = test_x[:n].to(DEVICE)

    _, cache_tr = model.run_with_cache(tr)
    _, cache_te = model.run_with_cache(te)

    key = f"blocks.{layer}.hook_resid_post"
    resid_tr = cache_tr[key].mean(1)
    resid_te = cache_te[key].mean(1)

    resid_tr = resid_tr - resid_tr.mean(0, keepdim=True)
    resid_te = resid_te - resid_te.mean(0, keepdim=True)

    contrast = resid_tr.mean(0) - resid_te.mean(0)
    _, _, V = torch.svd_lowrank(resid_tr, q=k + 4)
    basis = V[:, :k].T.clone()
    basis[0] = contrast
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    return basis

@torch.no_grad()
def make_random_subspace(k=K_DIMS, d=D_MODEL):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T

# ───────────────────────── Phase-cancellation hook ─────────────────────────
def make_phase_cancel_hook(B, coeff=2.0):
    B = B.detach()
    def hook_fn(resid, hook):
        flat = resid.reshape(-1, resid.shape[-1])
        proj = (flat @ B.T) @ B
        flat = flat - coeff * proj
        return flat.reshape(resid.shape)
    return hook_fn

# ───────────────────────── Evaluation ─────────────────────────
@torch.no_grad()
def evaluate(model, test_x, test_y, hook=None, layer=LAYER):
    model.eval()
    if hook is not None:
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]):
            logits = model(test_x.to(DEVICE))
    else:
        logits = model(test_x.to(DEVICE))
    loss = F.cross_entropy(logits[:, -1], test_y.to(DEVICE))
    acc = (logits[:, -1].argmax(-1) == test_y.to(DEVICE)).float().mean().item()
    return loss.item(), acc

# ───────────────────────── Main experiment ─────────────────────────
def run_one_seed(seed, alpha=1.0, max_steps=MAX_STEPS):
    """
    alpha=1.0 → no training-time suppression (pure post-grokking test)
    alpha=0.35 → optional training-time suppression
    """
    print(f"\n===== Seed {seed} | d_model={D_MODEL} | alpha={alpha} =====")
    torch.manual_seed(seed)

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed=seed)

    model = make_model(seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

    # Optional: extract early subspace for training-time suppression
    mem_B_early = None
    if alpha < 1.0:
        # quick warm-up to get a usable early subspace
        for _ in range(500):
            idx = torch.randint(0, len(train_x), (512,))
            logits = model(train_x[idx].to(DEVICE))
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        mem_B_early = extract_mem_subspace(model, train_x, test_x)

    # Train until grokking
    for step in range(max_steps):
        model.train()
        idx = torch.randint(0, len(train_x), (512,))
        xb, yb = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)

        if alpha < 1.0 and mem_B_early is not None:
            # apply training-time suppression
            def suppress_hook(resid, hook):
                flat = resid.reshape(-1, resid.shape[-1])
                proj = (flat @ mem_B_early.T) @ mem_B_early
                flat = flat - (1 - alpha) * proj
                return flat.reshape(resid.shape)
            with model.hooks(fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", suppress_hook)]):
                logits = model(xb)
        else:
            logits = model(xb)

        loss = F.cross_entropy(logits[:, -1], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0:
            tl, ta = evaluate(model, test_x, test_y)
            print(f"  step {step:5d}  test_loss={tl:.3f}  acc={ta:.3f}")
            if tl < 0.5:
                print("  → already grokked, stopping early")
                break

    # Final baseline
    base_loss, base_acc = evaluate(model, test_x, test_y)
    print(f"  Baseline: loss={base_loss:.4f}  acc={base_acc:.3f}")
    if base_loss > 1.0:
        print("  Did not grok — skipping ablation")
        return None

    # Extract subspace from the final (grokked) model
    mem_B = extract_mem_subspace(model, train_x, test_x)
    rand_B = make_random_subspace()

    results = {
        "seed": seed,
        "d_model": D_MODEL,
        "alpha": alpha,
        "baseline_loss": base_loss,
        "baseline_acc": base_acc,
        "coeffs": {}
    }

    for name, B in [("mem", mem_B), ("random", rand_B)]:
        results["coeffs"][name] = {}
        for coeff in [0.0, 1.0, 1.5, 2.0]:
            hook = make_phase_cancel_hook(B, coeff=coeff)
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            results["coeffs"][name][str(coeff)] = {"loss": loss, "acc": acc}
            print(f"  {name:6s} coeff={coeff:.1f} → loss={loss:.4f}  acc={acc:.3f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Training-time suppression (1.0 = none, 0.35 = Goldilocks value)")
    parser.add_argument("--out_dir", type=str, default="results_256")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)
    seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [8000, 8001, 8002, 8003])

    all_results = []
    for s in seeds:
        r = run_one_seed(s, alpha=args.alpha)
        if r is not None:
            all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"phase_256_alpha{args.alpha}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
