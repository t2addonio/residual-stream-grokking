#!/usr/bin/env python3
"""
train_dyck.py

Full experiment:
1. Train a small transformer on Dyck language (balanced parentheses)
2. Extract contrastive residual component (train vs held-out)
3. Extract algorithmic features (stack depth + counting)
4. Phase-cancel both + random control
5. Measure causal effect and orthogonality
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
import random
import math
from pathlib import Path
from datetime import datetime
import json

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
N_CTX = 32
VOCAB_SIZE = 3          # 0 = (, 1 = ), 2 = PAD
LR = 1e-3
WEIGHT_DECAY = 1.0
BATCH_SIZE = 64
MAX_STEPS = 30000
EVAL_EVERY = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_DIMS = 8              # rank of subspaces
LAYER = 1               # residual stream layer to analyze
SEED = 42

RESULTS_DIR = Path("results_dyck")
RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Data: Dyck language
# ------------------------------------------------------------------
OPEN, CLOSE, PAD = 0, 1, 2

def is_balanced(seq):
    depth = 0
    for t in seq:
        if t == PAD:
            break
        depth += 1 if t == OPEN else -1
        if depth < 0:
            return False
    return depth == 0

def generate_balanced(max_len, max_depth):
    for _ in range(100):
        seq = []
        depth = 0
        for _ in range(max_len):
            if depth == 0:
                choice = OPEN
            elif depth >= max_depth or len(seq) >= max_len - depth:
                choice = CLOSE
            else:
                choice = random.choice([OPEN, CLOSE])
            seq.append(choice)
            depth += 1 if choice == OPEN else -1
            if depth < 0:
                break
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

def get_train_heldout():
    # Train: shorter / shallower
    x_tr, y_tr = make_dataset(8000, max_len=16, max_depth=3, seed=SEED)
    # Held-out: longer / deeper (generalization)
    x_ho, y_ho = make_dataset(2000, max_len=28, max_depth=5, seed=SEED + 1)
    return (x_tr, y_tr), (x_ho, y_ho)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        act_fn="relu",
        normalization_type=None,
        d_vocab=VOCAB_SIZE,
        n_ctx=N_CTX,
        device=DEVICE,
    )
    return HookedTransformer(cfg)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def get_residuals(model, x, layer=LAYER):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        return cache[f"blocks.{layer}.hook_resid_post"].detach()

def extract_contrastive_subspace(model, x_train, x_held, k=K_DIMS, layer=LAYER):
    r_tr = get_residuals(model, x_train, layer=layer)
    r_ho = get_residuals(model, x_held, layer=layer)
    # mean over batch and sequence
    mu_tr = r_tr.mean(dim=(0, 1))
    mu_ho = r_ho.mean(dim=(0, 1))
    diff = mu_tr - mu_ho
    flat = r_tr.reshape(-1, D_MODEL) - mu_tr
    flat = torch.cat([diff.unsqueeze(0), flat], dim=0)
    _, _, Vh = torch.linalg.svd(flat.float(), full_matrices=False)
    B = Vh[:k]
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B

def extract_algorithmic_features(model, x, y, k=K_DIMS, layer=LAYER):
    """
    Simple algorithmic features for Dyck:
    - Stack depth at each position
    - Running open-count and close-count
    We fit a linear map from residual → these features and take principal directions.
    """
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # [B, S, D]

    # Build targets: stack depth, open count, close count
    B, S, D = resid.shape
    targets = []
    for b in range(B):
        depth = 0
        open_c = 0
        close_c = 0
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
    targets = torch.tensor(targets, dtype=torch.float32, device=DEVICE)  # [B, S, 3]

    resid_flat = resid.reshape(-1, D)
    targets_flat = targets.reshape(-1, 3)
    # center
    resid_flat = resid_flat - resid_flat.mean(0, keepdim=True)
    targets_flat = targets_flat - targets_flat.mean(0, keepdim=True)

    W = torch.linalg.lstsq(resid_flat, targets_flat).solution  # [D, 3]
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    B_algo = U[:, :k].T
    B_algo = torch.linalg.qr(B_algo.T, mode="reduced")[0].T
    return B_algo

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
        # use final non-pad position roughly
        preds = logits[:, -1].argmax(-1)
        # For simplicity we treat the model as classifying the whole sequence
        # via the final token prediction (you can improve this later)
        acc = (preds % 2 == y.to(DEVICE)).float().mean().item()  # crude but works for binary
    model.reset_hooks()
    return acc

def subspace_overlap(A, B):
    """Sum of squared cosines of principal angles (0 = orthogonal, k = identical)."""
    A = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    B = B / (B.norm(dim=1, keepdim=True) + 1e-8)
    M = A @ B.T
    return (M ** 2).sum().item()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print("Building data...")
    (x_tr, y_tr), (x_ho, y_ho) = get_train_heldout()
    x_tr, y_tr = x_tr.to(DEVICE), y_tr.to(DEVICE)
    x_ho, y_ho = x_ho.to(DEVICE), y_ho.to(DEVICE)
    print(f"Train: {x_tr.shape}, Held-out: {x_ho.shape}")

    print("Making model...")
    model = make_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("Training...")
    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(x_tr), (BATCH_SIZE,), device=DEVICE)
        xb, yb = x_tr[idx], y_tr[idx]
        logits = model(xb)
        # simple classification loss on last token
        loss = F.cross_entropy(logits[:, -1], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            acc_tr = evaluate(model, x_tr[:1024], y_tr[:1024])
            acc_ho = evaluate(model, x_ho[:1024], y_ho[:1024])
            print(f"step {step:5d}  loss {loss.item():.4f}  train_acc {acc_tr:.3f}  held_acc {acc_ho:.3f}")
            if acc_ho > 0.90 and step > 5000:
                print("→ generalization reached, stopping early")
                break

    # Save checkpoint
    ckpt_path = RESULTS_DIR / "dyck_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model → {ckpt_path}")

    # --------------------------------------------------------------
    # Extract subspaces
    # --------------------------------------------------------------
    print("\nExtracting contrastive residual subspace...")
    B_contrast = extract_contrastive_subspace(model, x_tr[:2048], x_ho[:2048])
    print(f"Contrastive B shape: {B_contrast.shape}")

    print("Extracting algorithmic (stack/counting) subspace...")
    B_algo = extract_algorithmic_features(model, x_tr[:1024], y_tr[:1024])
    print(f"Algorithmic B shape: {B_algo.shape}")

    B_rand = torch.linalg.qr(torch.randn(K_DIMS, D_MODEL, device=DEVICE).T, mode="reduced")[0].T

    # --------------------------------------------------------------
    # Causal evaluation
    # --------------------------------------------------------------
    print("\n=== Causal evaluation on held-out set ===")
    acc_base = evaluate(model, x_ho, y_ho)
    print(f"Baseline held-out acc: {acc_base:.3f}")

    for name, B in [("contrastive", B_contrast), ("algorithmic", B_algo), ("random", B_rand)]:
        acc = evaluate(model, x_ho, y_ho, hook=make_cancel_hook(B, coeff=2.0))
        print(f"Cancel {name:12s} (coeff=2): {acc:.3f}")

    # Orthogonality
    overlap = subspace_overlap(B_contrast, B_algo)
    print(f"\nOverlap (contrastive vs algorithmic): {overlap:.4f}  (0 = orthogonal, {K_DIMS} = identical)")

    # Save everything
    results = {
        "acc_base": acc_base,
        "overlap_contrastive_algo": overlap,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "B_contrast": B_contrast.cpu(),
        "B_algo": B_algo.cpu(),
        "B_rand": B_rand.cpu(),
    }, RESULTS_DIR / "subspaces.pt")
    print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
