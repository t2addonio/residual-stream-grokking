#!/usr/bin/env python3
"""
run_full_256.py

Full hypothesis test at d_model=256.

Supports:
  - Training-time directional suppression (Goldilocks α-sweep)
  - Post-grokking phase cancellation + random control
  - Recovery test
  - Clean parallel execution (each process writes its own file)

Usage examples:
  # Single seed, Goldilocks α
  python run_full_256.py --condition mem_suppress --alpha 0.35 --start 9000 --end 9000

  # 15 seeds of control
  python run_full_256.py --condition control --start 9100 --end 9114

  # Full post-grokking ablation on a seed that already grokked
  python run_full_256.py --condition ablation --alpha 0.35 --start 9000 --end 9000
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ───────────────────────── Config (d_model = 256) ─────────────────────────
P = 113
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
K_DIMS = 16
LAYER = 1
MAX_STEPS = 25001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

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

# ───────────────────────── Subspace ─────────────────────────
@torch.no_grad()
def extract_mem_subspace(model, train_x, test_x, layer=LAYER, k=K_DIMS):
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
def make_random_subspace(k=K_DIMS, d=D_MODEL):
    A = torch.randn(k, d, device=DEVICE)
    Q, _ = torch.linalg.qr(A.T, mode="reduced")
    return Q.T

# ───────────────────────── Hooks ─────────────────────────
def make_suppress_hook(B, alpha):
    B = B.detach()
    def hook_fn(resid, hook):
        flat = resid.reshape(-1, resid.shape[-1])
        proj = (flat @ B.T) @ B
        flat = flat - (1.0 - alpha) * proj
        return flat.reshape(resid.shape)
    return hook_fn

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

# ───────────────────────── Core runner ─────────────────────────
def run_one(seed, condition="mem_suppress", alpha=0.35, do_ablation=True):
    print(f"\n===== Seed {seed} | condition={condition} | alpha={alpha} | d_model={D_MODEL} =====")
    torch.manual_seed(seed)

    x, y = make_dataset()
    (train_x, train_y), (test_x, test_y) = split_dataset(x, y, seed=seed)
    model = make_model(seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

    # Early subspace for training-time suppression
    mem_B = None
    if condition == "mem_suppress" and alpha < 1.0:
        for _ in range(800):  # short warm-up
            idx = torch.randint(0, len(train_x), (BATCH_SIZE,))
            logits = model(train_x[idx].to(DEVICE))
            loss = F.cross_entropy(logits[:, -1], train_y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        mem_B = extract_mem_subspace(model, train_x, test_x)

    history = {"step": [], "train_loss": [], "test_loss": [], "test_acc": []}

    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(train_x), (BATCH_SIZE,))
        xb = train_x[idx].to(DEVICE)
        yb = train_y[idx].to(DEVICE)

        if condition == "mem_suppress" and mem_B is not None and alpha < 1.0:
            hook = make_suppress_hook(mem_B, alpha)
            with model.hooks(fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook)]):
                logits = model(xb)
        else:
            logits = model(xb)

        loss = F.cross_entropy(logits[:, -1], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == MAX_STEPS - 1:
            tl, ta = evaluate(model, test_x, test_y)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(tl)
            history["test_acc"].append(ta)
            print(f"  step {step:5d}  test_loss={tl:.3f}  acc={ta:.3f}")
            if tl < 0.5:
                print("  → grokked")
                break

    final_loss, final_acc = evaluate(model, test_x, test_y)
    grokked = final_loss < 0.8

    result = {
        "seed": seed,
        "condition": condition,
        "alpha": alpha,
        "d_model": D_MODEL,
        "final_test_loss": final_loss,
        "final_test_acc": final_acc,
        "grokked": grokked,
        "history": history,
        "ablation": None,
    }

    # Post-grokking phase cancellation (the key causal test)
    if do_ablation and grokked:
        print("  Running post-grokking phase cancellation...")
        mem_B_final = extract_mem_subspace(model, train_x, test_x)
        rand_B = make_random_subspace()

        ablation = {"mem": {}, "random": {}, "recovery": {}}

        for name, B in [("mem", mem_B_final), ("random", rand_B)]:
            for coeff in [0.0, 1.0, 2.0]:
                hook = make_phase_cancel_hook(B, coeff=coeff)
                loss, acc = evaluate(model, test_x, test_y, hook=hook)
                ablation[name][str(coeff)] = {"loss": loss, "acc": acc}
                print(f"    {name:6s} c={coeff:.1f} → loss={loss:.4f}  acc={acc:.3f}")

        # Recovery test
        loss_r, acc_r = evaluate(model, test_x, test_y, hook=None)
        ablation["recovery"] = {"loss": loss_r, "acc": acc_r}
        print(f"    recovery     → loss={loss_r:.4f}  acc={acc_r:.3f}")

        result["ablation"] = ablation

    return result

# ───────────────────────── Main ─────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="mem_suppress",
                        choices=["mem_suppress", "control"])
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results_256")
    parser.add_argument("--no_ablation", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    if args.condition == "control":
        args.alpha = 1.0

    all_results = []
    for seed in range(args.start, args.end + 1):
        r = run_one(
            seed=seed,
            condition=args.condition,
            alpha=args.alpha,
            do_ablation=not args.no_ablation,
        )
        all_results.append(r)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.condition}_a{args.alpha}_{args.start}_{args.end}_{stamp}.json"
    out_path = Path(args.out_dir) / fname
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
