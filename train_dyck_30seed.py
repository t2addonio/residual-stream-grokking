#!/usr/bin/env python3
"""
train_dyck_30seed.py

30-seed Dyck language experiment.
Designed to be launched one seed per GPU on an 8×4090 box.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
import random
from pathlib import Path
from datetime import datetime
import json
import argparse

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
LR = 1e-3
WEIGHT_DECAY = 1.0
BATCH_SIZE = 64
MAX_STEPS = 30000
EVAL_EVERY = 1000
K_DIMS = 8
LAYER = 1
BASE_SEED = 100

RESULTS_DIR = Path("results_dyck_30seed")

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
OPEN, CLOSE, PAD = 0, 1, 2

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

def get_train_heldout(seed):
    x_tr, y_tr = make_dataset(8000, max_len=16, max_depth=3, seed=seed)
    x_ho, y_ho = make_dataset(2000, max_len=28, max_depth=5, seed=seed + 10000)
    # make_dataset emits all positives first, then all negatives. Shuffle the
    # held-out set (seeded) so any prefix of it is a fair sample.
    g = torch.Generator().manual_seed(seed + 20000)
    perm = torch.randperm(len(y_ho), generator=g)
    return (x_tr, y_tr), (x_ho[perm], y_ho[perm])

def balanced_subset(x, y, n_per_class, seed):
    """Seeded subset with exactly n_per_class examples of each label."""
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in (0, 1):
        cls = (y == c).nonzero(as_tuple=True)[0].cpu()
        cls = cls[torch.randperm(len(cls), generator=g)][:n_per_class]
        idx.append(cls)
    idx = torch.cat(idx)
    idx = idx[torch.randperm(len(idx), generator=g)].to(x.device)
    return x[idx], y[idx]

# ------------------------------------------------------------------
# Model + helpers
# ------------------------------------------------------------------
def make_model(device):
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_head=D_HEAD, d_mlp=D_MLP, act_fn="relu",
        normalization_type=None, d_vocab=VOCAB_SIZE, n_ctx=N_CTX, device=device,
    )
    return HookedTransformer(cfg)

def get_residuals(model, x, layer=LAYER):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x)
        return cache[f"blocks.{layer}.hook_resid_post"].detach()

def extract_contrastive_subspace(model, x_train, x_held, k=K_DIMS, layer=LAYER):
    r_tr = get_residuals(model, x_train, layer=layer)
    r_ho = get_residuals(model, x_held, layer=layer)
    mu_tr = r_tr.mean(dim=(0, 1))
    mu_ho = r_ho.mean(dim=(0, 1))
    diff = mu_tr - mu_ho
    flat = r_tr.reshape(-1, D_MODEL) - mu_tr
    flat = torch.cat([diff.unsqueeze(0), flat], dim=0)
    _, _, Vh = torch.linalg.svd(flat.float(), full_matrices=False)
    B = Vh[:k]
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B

def extract_algorithmic_features(model, x, k=K_DIMS, layer=LAYER):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x)
        resid = cache[f"blocks.{layer}.hook_resid_post"]
    Bsz, S, D = resid.shape
    targets = []
    for b in range(Bsz):
        depth = open_c = close_c = 0
        seq_feats = []
        for t in x[b]:
            t = t.item()
            if t == OPEN:
                open_c += 1
                depth += 1
            elif t == CLOSE:
                close_c += 1
                depth = max(0, depth - 1)
            seq_feats.append([depth, open_c, close_c])
        targets.append(seq_feats)
    targets = torch.tensor(targets, dtype=torch.float32, device=x.device)
    resid_flat = resid.reshape(-1, D) - resid.reshape(-1, D).mean(0, keepdim=True)
    targets_flat = targets.reshape(-1, 3) - targets.reshape(-1, 3).mean(0, keepdim=True)
    W = torch.linalg.lstsq(resid_flat, targets_flat).solution
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    B_algo = U[:, :k].T
    B_algo = torch.linalg.qr(B_algo.T, mode="reduced")[0].T
    return B_algo

def make_cancel_hook(B, coeff=2.0):
    def hook(resid, hook):
        proj = torch.einsum("bpd,kd->bpk", resid, B)
        delta = torch.einsum("bpk,kd->bpd", proj, B)
        return resid - coeff * delta
    return hook

def _final_logits(model, x, hook=None):
    model.eval()
    model.reset_hooks()
    if hook is not None:
        model.add_hook(f"blocks.{LAYER}.hook_resid_post", hook)
    with torch.no_grad():
        logits = model(x)[:, -1]
    model.reset_hooks()
    return logits

def evaluate(model, x, y, hook=None):
    """Binary accuracy from the two class logits (token ids 0/1).

    The old metric was ``argmax over all 3 vocab logits % 2 == y``. That let the PAD
    logit (id 2, and 2 % 2 == 0) count as "predict 0", so an intervention that made PAD
    win turned into a constant class-0 prediction (acc 0.500) instead of showing up as a
    sign flip. We now compare only the two class logits.
    """
    preds = _final_logits(model, x, hook)[:, :2].argmax(-1)
    return (preds == y).float().mean().item()

def prediction_stats(model, x, y, hook=None):
    """Extra diagnostics so label flips and single-class collapse can be told apart."""
    logits = _final_logits(model, x, hook)
    preds = logits[:, :2].argmax(-1)
    return {
        "acc": (preds == y).float().mean().item(),
        "pred_pos_rate": preds.float().mean().item(),
        "acc_pos": (preds[y == 1] == 1).float().mean().item(),
        "acc_neg": (preds[y == 0] == 0).float().mean().item(),
        "pad_argmax_rate": (logits.argmax(-1) == PAD).float().mean().item(),
    }

def subspace_overlap(A, B, rtol=1e-6):
    """Sum of squared cosines of principal angles / min(rank), in [0, 1].

    Builds a numerical orthonormal basis for each row-span (via SVD) before
    measuring principal angles. Row-normalized-but-correlated directions can
    no longer push the score above 1 (the bug that made Dyck
    overlap_contrastive_algo report values like 2.7).
    """
    def _basis(X):
        # X: (k, d) rows spanning the subspace -> (rank, d) orthonormal rows
        _, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)
        if S.numel() == 0:
            return X[:0]
        thresh = float(S[0]) * rtol + 1e-12
        rank = int((S > thresh).sum().item())
        rank = max(rank, 1)
        return Vh[:rank]

    A_q, B_q = _basis(A), _basis(B)
    if A_q.numel() == 0 or B_q.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(A_q @ B_q.T)
    return (s ** 2).sum().item() / min(A_q.shape[0], B_q.shape[0])


# ------------------------------------------------------------------
# Single seed
# ------------------------------------------------------------------
def readout_alignment(model, B):
    """Fraction of the class logit-difference direction (W_U[:,1]-W_U[:,0]) inside span(B)."""
    d = (model.W_U[:, 1] - model.W_U[:, 0]).detach()
    return ((B @ d).norm() / (d.norm() + 1e-12)).item()

def run_seed(seed, device, max_steps=MAX_STEPS, min_steps=5000, results_dir=RESULTS_DIR):
    print(f"\n{'='*60}")
    print(f"SEED {seed} on {device}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    random.seed(seed)

    (x_tr, y_tr), (x_ho, y_ho) = get_train_heldout(seed)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_ho, y_ho = x_ho.to(device), y_ho.to(device)

    model = make_model(device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Balanced early-stop set: 512 per class from the (shuffled) held-out set.
    x_es, y_es = balanced_subset(x_ho, y_ho, 512, seed + 30000)

    for step in range(max_steps):
        model.train()
        idx = torch.randint(0, len(x_tr), (BATCH_SIZE,), device=device)
        logits = model(x_tr[idx])
        loss = F.cross_entropy(logits[:, -1], y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == max_steps - 1:
            acc_ho = evaluate(model, x_es, y_es)
            print(f"  step {step:5d}  loss {loss.item():.4f}  held_acc {acc_ho:.3f}")
            if acc_ho > 0.90 and step > min_steps:
                print("  → generalization reached")
                break

    # Extract subspaces
    B_contrast = extract_contrastive_subspace(model, x_tr[:2048], x_ho[:2048])
    B_algo = extract_algorithmic_features(model, x_tr[:1024])
    g_rand = torch.Generator().manual_seed(seed + 40000)
    B_rand = torch.linalg.qr(torch.randn(K_DIMS, D_MODEL, generator=g_rand).T.to(device), mode="reduced")[0].T

    # Causal evaluation
    acc_base = evaluate(model, x_ho, y_ho)
    acc_contrast = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_contrast, coeff=2.0))
    acc_algo = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_algo, coeff=2.0))
    acc_rand = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_rand, coeff=2.0))
    overlap = subspace_overlap(B_contrast, B_algo)
    hooks = {"base": None,
             "contrastive": make_cancel_hook(B_contrast, coeff=2.0),
             "algorithmic": make_cancel_hook(B_algo, coeff=2.0),
             "random": make_cancel_hook(B_rand, coeff=2.0)}
    stats = {k: prediction_stats(model, x_ho, y_ho, hook=h) for k, h in hooks.items()}

    print("\n=== Causal evaluation ===")
    print(f"Baseline:              {acc_base:.3f}")
    print(f"Cancel contrastive:    {acc_contrast:.3f}")
    print(f"Cancel algorithmic:    {acc_algo:.3f}")
    print(f"Cancel random:         {acc_rand:.3f}")
    print(f"Overlap:               {overlap:.4f}")
    for k, st in stats.items():
        print(f"  {k:<12} acc {st['acc']:.3f}  pred_pos_rate {st['pred_pos_rate']:.3f}  "
              f"pad_argmax {st['pad_argmax_rate']:.3f}")

    results = {
        "seed": seed,
        "acc_base": acc_base,
        "acc_cancel_contrastive": acc_contrast,
        "acc_cancel_algorithmic": acc_algo,
        "acc_cancel_random": acc_rand,
        "overlap_contrastive_algo": overlap,
        # extract_algorithmic_features regresses on 3 targets, so its subspace has
        # rank <= 3 even though K_DIMS=8 is requested.
        "k_contrastive": int(B_contrast.shape[0]),
        "k_algorithmic": int(B_algo.shape[0]),
        "k_random": int(B_rand.shape[0]),
        "readout_alignment": {"contrastive": readout_alignment(model, B_contrast),
                              "algorithmic": readout_alignment(model, B_algo),
                              "random": readout_alignment(model, B_rand)},
        "prediction_stats": stats,
        "steps_trained": step + 1,
        "metric": "argmax over class logits {0,1}; held-out shuffled; balanced early stop",
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Save
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"seed_{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "model_state": model.state_dict(),
        "B_contrast": B_contrast.cpu(),
        "B_algo": B_algo.cpu(),
        "B_rand": B_rand.cpu(),
    }, results_dir / f"seed_{seed}.pt")

    return results

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--min_steps", type=int, default=5000,
                        help="don't early-stop before this many steps")
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        device = "cpu"
        print("CUDA not available (or --cpu): running on CPU")
    run_seed(args.seed, device, max_steps=args.max_steps, min_steps=args.min_steps,
             results_dir=args.results_dir)
