#!/usr/bin/env python3
"""
analyze_dyck_directions.py

Direction-level analysis of the Dyck contrastive residual component
(mirrors the sparse-parity 2-D core analysis).

For each seed (or a chosen subset):
1. Singular spectrum of the contrastive subspace
2. Max-activating sequences
3. Linear probes (stack depth, open/close counts, balance)
4. Phase-cancellation of top-1 and top-2 core only
5. Comparison against the algorithmic (stack/counting) subspace
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
import random
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
N_CTX = 32
VOCAB_SIZE = 3
LAYER = 1
K_DIMS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path("results_dyck_30seed")
OUT_DIR = Path("analysis_dyck_directions")
OUT_DIR.mkdir(exist_ok=True)

OPEN, CLOSE, PAD = 0, 1, 2

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_head=D_HEAD, d_mlp=D_MLP, act_fn="relu",
        normalization_type=None, d_vocab=VOCAB_SIZE, n_ctx=N_CTX, device=DEVICE,
    )
    return HookedTransformer(cfg)

# ------------------------------------------------------------------
# Data helpers (same as training)
# ------------------------------------------------------------------
def is_balanced(seq):
    depth = 0
    for t in seq:
        if t == PAD: break
        depth += 1 if t == OPEN else -1
        if depth < 0: return False
    return depth == 0

def generate_balanced(max_len, max_depth):
    for _ in range(100):
        seq, depth = [], 0
        for _ in range(max_len):
            if depth == 0:
                choice = OPEN
            elif depth >= max_depth or len(seq) >= max_len - depth:
                choice = CLOSE
            else:
                choice = random.choice([OPEN, CLOSE])
            seq.append(choice)
            depth += 1 if choice == OPEN else -1
            if depth < 0: break
        if depth == 0 and len(seq) >= 2:
            return seq
    return [OPEN, CLOSE]

def generate_unbalanced(max_len):
    for _ in range(100):
        length = random.randint(2, max_len)
        seq = [random.choice([OPEN, CLOSE]) for _ in range(length)]
        if not is_balanced(seq):
            return seq
    return [OPEN, OPEN, CLOSE]

def make_dataset(n, max_len, max_depth, seed, balanced_ratio=0.5):
    random.seed(seed)
    xs, ys = [], []
    n_bal = int(n * balanced_ratio)
    for i in range(n):
        if i < n_bal:
            seq = generate_balanced(max_len, max_depth)
            label = 1
        else:
            seq = generate_unbalanced(max_len)
            label = 0
        padded = seq + [PAD] * (max_len - len(seq))
        xs.append(padded[:max_len])
        ys.append(label)
    return torch.tensor(xs), torch.tensor(ys)

def seq_to_str(seq):
    chars = []
    for t in seq:
        if t == OPEN: chars.append("(")
        elif t == CLOSE: chars.append(")")
        else: break
    return "".join(chars)

# ------------------------------------------------------------------
# Core analysis functions
# ------------------------------------------------------------------
@torch.no_grad()
def get_residuals(model, x, layer=LAYER):
    model.eval()
    _, cache = model.run_with_cache(x.to(DEVICE))
    return cache[f"blocks.{layer}.hook_resid_post"]  # [B, S, D]

def singular_spectrum(B):
    """Return normalized singular values of the subspace."""
    # B is already [k, d] orthonormal
    # We look at the energy of the contrastive extraction itself if available,
    # otherwise just report the rank structure.
    return torch.ones(B.shape[0])  # placeholder; real spectrum comes from the SVD that built B

def max_activating_sequences(model, B, n_samples=2000, top_k=15, seed=0):
    """Find sequences that most strongly activate the contrastive directions."""
    x, y = make_dataset(n_samples, max_len=28, max_depth=5, seed=seed)
    x = x.to(DEVICE)
    resid = get_residuals(model, x)                     # [B, S, D]
    # Project onto each direction of B
    # score = mean absolute projection across sequence
    proj = torch.einsum("bsd,kd->bsk", resid, B.to(DEVICE))  # [B, S, k]
    scores = proj.abs().mean(dim=1)                     # [B, k]
    total_score = scores.sum(dim=1)                     # [B]

    top_idx = total_score.argsort(descending=True)[:top_k]
    results = []
    for i in top_idx:
        results.append({
            "seq": seq_to_str(x[i].cpu().tolist()),
            "label": int(y[i]),
            "score": float(total_score[i]),
            "per_dir": scores[i].cpu().tolist(),
        })
    return results

def linear_probes(model, B, n_samples=1500, seed=1):
    """Probe residual and contrastive projection for stack features."""
    x, y = make_dataset(n_samples, max_len=24, max_depth=5, seed=seed)
    x = x.to(DEVICE)
    resid = get_residuals(model, x)                     # [B, S, D]
    B = B.to(DEVICE)

    # Build targets per position
    depths, opens, closes, balances = [], [], [], []
    for b in range(len(x)):
        depth = open_c = close_c = 0
        for t in x[b]:
            t = t.item()
            if t == OPEN:
                open_c += 1
                depth += 1
            elif t == CLOSE:
                close_c += 1
                depth = max(0, depth - 1)
            depths.append(depth)
            opens.append(open_c)
            closes.append(close_c)
            balances.append(1 if depth == 0 else 0)

    targets = {
        "depth": torch.tensor(depths, dtype=torch.float32, device=DEVICE),
        "open": torch.tensor(opens, dtype=torch.float32, device=DEVICE),
        "close": torch.tensor(closes, dtype=torch.float32, device=DEVICE),
        "balance": torch.tensor(balances, dtype=torch.float32, device=DEVICE),
    }

    resid_flat = resid.reshape(-1, D_MODEL)
    proj_flat = torch.einsum("bd,kd->bk", resid_flat, B)  # projection onto contrastive subspace

    def fit_r2(features, target):
        features = features - features.mean(0, keepdim=True)
        target = target - target.mean()
        # simple least-squares
        W = torch.linalg.lstsq(features, target.unsqueeze(1)).solution
        pred = features @ W
        ss_res = ((target - pred.squeeze()) ** 2).sum()
        ss_tot = (target ** 2).sum() + 1e-8
        return float(1 - ss_res / ss_tot)

    results = {}
    for name, tgt in targets.items():
        results[f"resid_{name}_R2"] = fit_r2(resid_flat, tgt)
        results[f"contrast_{name}_R2"] = fit_r2(proj_flat, tgt)
    return results

def make_cancel_hook(B, coeff=2.0):
    B = B.to(DEVICE)
    def hook(resid, hook):
        proj = torch.einsum("bpd,kd->bpk", resid, B)
        delta = torch.einsum("bpk,kd->bpd", proj, B)
        return resid - coeff * delta
    return hook

def evaluate(model, x, y, hook=None):
    model.eval()
    model.reset_hooks()
    if hook is not None:
        model.add_hook(f"blocks.{LAYER}.hook_resid_post", hook)
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        preds = logits[:, -1].argmax(-1)
        acc = (preds % 2 == y.to(DEVICE)).float().mean().item()
    model.reset_hooks()
    return acc

def core_cancellation_test(model, B_contrast, x_ho, y_ho):
    """Test whether top-1 or top-2 directions are already sufficient for collapse."""
    results = {}
    # full subspace
    results["full_k8"] = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_contrast, 2.0))
    # top-1
    B1 = B_contrast[:1]
    results["top1"] = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B1, 2.0))
    # top-2
    B2 = B_contrast[:2]
    results["top2"] = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B2, 2.0))
    # top-4
    B4 = B_contrast[:4]
    results["top4"] = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B4, 2.0))
    return results

# ------------------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------------------
def analyze_seed(seed):
    print(f"\n{'='*60}")
    print(f"ANALYZING SEED {seed}")
    print(f"{'='*60}")

    pt_path = RESULTS_DIR / f"seed_{seed}.pt"
    if not pt_path.exists():
        print(f"  missing {pt_path}, skipping")
        return None

    data = torch.load(pt_path, map_location="cpu")
    B_contrast = data["B_contrast"]
    B_algo = data["B_algo"]

    model = make_model().to(DEVICE)
    model.load_state_dict(data["model_state"])
    model.eval()

    # Held-out set for causal tests
    x_ho, y_ho = make_dataset(2000, max_len=28, max_depth=5, seed=seed + 9999)
    x_ho, y_ho = x_ho.to(DEVICE), y_ho.to(DEVICE)
    # rebuild properly
    x_ho, y_ho = make_dataset(2000, max_len=28, max_depth=5, seed=seed + 9999)
    x_ho, y_ho = x_ho.to(DEVICE), y_ho.to(DEVICE)

    # 1. Baseline & full cancellation (sanity)
    acc_base = evaluate(model, x_ho, y_ho)
    acc_full = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_contrast, 2.0))
    print(f"  Baseline:     {acc_base:.3f}")
    print(f"  Full cancel:  {acc_full:.3f}")

    # 2. Core cancellation
    core = core_cancellation_test(model, B_contrast, x_ho, y_ho)
    print(f"  Core cancel  top1={core['top1']:.3f}  top2={core['top2']:.3f}  top4={core['top4']:.3f}")

    # 3. Max-activating sequences
    max_acts = max_activating_sequences(model, B_contrast, n_samples=2000, top_k=12, seed=seed)
    print("  Top activating sequences:")
    for m in max_acts[:5]:
        print(f"    {m['seq']:<30}  label={m['label']}  score={m['score']:.3f}")

    # 4. Linear probes
    probes = linear_probes(model, B_contrast, n_samples=1500, seed=seed+1)
    print("  Probe R²:")
    for k, v in probes.items():
        print(f"    {k}: {v:.3f}")

    # 5. Overlap with algorithmic features (already known, but re-confirm)
    A = B_contrast / (B_contrast.norm(dim=1, keepdim=True) + 1e-8)
    Al = B_algo / (B_algo.norm(dim=1, keepdim=True) + 1e-8)
    overlap = ((A @ Al.T) ** 2).sum().item()
    print(f"  Overlap vs algorithmic: {overlap:.4f}")

    out = {
        "seed": seed,
        "acc_base": acc_base,
        "acc_full_cancel": acc_full,
        "core_cancel": core,
        "max_activating": max_acts,
        "probes": probes,
        "overlap_algo": overlap,
    }

    with open(OUT_DIR / f"analysis_seed_{seed}.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

# ------------------------------------------------------------------
if __name__ == "__main__":
    # Analyze all seeds that finished (or a subset)
    seed_files = sorted(RESULTS_DIR.glob("seed_*.pt"))
    seeds = [int(p.stem.split("_")[1]) for p in seed_files]
    print(f"Found {len(seeds)} finished seeds: {seeds}")

    all_results = []
    for seed in seeds:
        res = analyze_seed(seed)
        if res is not None:
            all_results.append(res)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY – Core cancellation sufficiency")
    print("="*70)
    print(f"{'Seed':<8} {'Base':>7} {'Full':>7} {'Top1':>7} {'Top2':>7} {'Top4':>7}")
    for r in all_results:
        c = r["core_cancel"]
        print(f"{r['seed']:<8} {r['acc_base']:>7.3f} {r['acc_full_cancel']:>7.3f} "
              f"{c['top1']:>7.3f} {c['top2']:>7.3f} {c['top4']:>7.3f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll analysis written to {OUT_DIR}/")
