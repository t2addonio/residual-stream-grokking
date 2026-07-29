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
RESULTS_DIR.mkdir(exist_ok=True)

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
    return (x_tr, y_tr), (x_ho, y_ho)

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

def evaluate(model, x, y, hook=None):
    model.eval()
    model.reset_hooks()
    if hook is not None:
        model.add_hook(f"blocks.{LAYER}.hook_resid_post", hook)
    with torch.no_grad():
        logits = model(x)
        preds = logits[:, -1].argmax(-1)
        acc = (preds % 2 == y).float().mean().item()
    model.reset_hooks()
    return acc

def subspace_overlap(A, B):
    A = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    B = B / (B.norm(dim=1, keepdim=True) + 1e-8)
    return ((A @ B.T) ** 2).sum().item()

# ------------------------------------------------------------------
# Single seed
# ------------------------------------------------------------------
def run_seed(seed, device):
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

    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(x_tr), (BATCH_SIZE,), device=device)
        logits = model(x_tr[idx])
        loss = F.cross_entropy(logits[:, -1], y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            acc_ho = evaluate(model, x_ho[:1024], y_ho[:1024])
            print(f"  step {step:5d}  loss {loss.item():.4f}  held_acc {acc_ho:.3f}")
            if acc_ho > 0.90 and step > 5000:
                print("  → generalization reached")
                break

    # Extract subspaces
    B_contrast = extract_contrastive_subspace(model, x_tr[:2048], x_ho[:2048])
    B_algo = extract_algorithmic_features(model, x_tr[:1024])
    B_rand = torch.linalg.qr(torch.randn(K_DIMS, D_MODEL, device=device).T, mode="reduced")[0].T

    # Causal evaluation
    acc_base = evaluate(model, x_ho, y_ho)
    acc_contrast = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_contrast, coeff=2.0))
    acc_algo = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_algo, coeff=2.0))
    acc_rand = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B_rand, coeff=2.0))
    overlap = subspace_overlap(B_contrast, B_algo)

    print("\n=== Causal evaluation ===")
    print(f"Baseline:              {acc_base:.3f}")
    print(f"Cancel contrastive:    {acc_contrast:.3f}")
    print(f"Cancel algorithmic:    {acc_algo:.3f}")
    print(f"Cancel random:         {acc_rand:.3f}")
    print(f"Overlap:               {overlap:.4f}")

    results = {
        "seed": seed,
        "acc_base": acc_base,
        "acc_cancel_contrastive": acc_contrast,
        "acc_cancel_algorithmic": acc_algo,
        "acc_cancel_random": acc_rand,
        "overlap_contrastive_algo": overlap,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Save
    with open(RESULTS_DIR / f"seed_{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "model_state": model.state_dict(),
        "B_contrast": B_contrast.cpu(),
        "B_algo": B_algo.cpu(),
        "B_rand": B_rand.cpu(),
    }, RESULTS_DIR / f"seed_{seed}.pt")

    return results

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    run_seed(args.seed, device)
