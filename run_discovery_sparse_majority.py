#!/usr/bin/env python3
"""run_discovery_sparse_majority.py"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

N_BITS = 20
K_SPARSE = 5          # majority of first 5 bits
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
K_DIMS = 16
LAYER = 1
MAX_STEPS = 40001
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0
N_TRAIN = 10000
N_TEST = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results_sparse_majority")
RESULTS_DIR.mkdir(exist_ok=True)

def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
        d_head=D_HEAD, d_mlp=D_MLP, act_fn="relu",
        normalization_type=None, d_vocab=2, n_ctx=N_BITS, device=DEVICE,
    )
    return HookedTransformer(cfg)

def make_sparse_majority_dataset(n_samples, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n_samples, N_BITS), generator=g)
    # majority of first K_SPARSE bits
    y = (x[:, :K_SPARSE].sum(dim=1) > (K_SPARSE // 2)).long()
    return x, y

def get_residuals(model, x, layer=LAYER):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        return cache[f"blocks.{layer}.hook_resid_post"].detach()

def extract_contrastive_subspace(model, train_x, test_x, k=K_DIMS):
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
    return B

def make_cancel_hook(B, coeff):
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
        loss = F.cross_entropy(logits[:, -1], y.to(DEVICE))
        acc = (logits[:, -1].argmax(-1) == y.to(DEVICE)).float().mean()
    model.reset_hooks()
    return loss.item(), acc.item()

def logit_contribution(model, x, y, B):
    model.eval()
    with torch.no_grad():
        logits, cache = model.run_with_cache(x.to(DEVICE))
        resid = cache[f"blocks.{LAYER}.hook_resid_post"]
        W_U = model.W_U
        correct = y.to(DEVICE)
        full_logit = logits[:, -1].gather(1, correct.unsqueeze(1)).squeeze(1)
        B = B.to(DEVICE)
        proj_coeff = torch.einsum("bpd,kd->bpk", resid, B)
        resid_proj = torch.einsum("bpk,kd->bpd", proj_coeff, B)
        logit_proj = (resid_proj[:, -1] @ W_U).gather(1, correct.unsqueeze(1)).squeeze(1)
        frac = (logit_proj / (full_logit + 1e-8)).mean().item()
    return frac

def run_one(seed):
    print(f"\n===== Seed {seed} | sparse majority n={N_BITS} k={K_SPARSE} =====", flush=True)
    torch.manual_seed(seed)
    model = make_model().to(DEVICE)
    train_x, train_y = make_sparse_majority_dataset(N_TRAIN, seed)
    test_x, test_y = make_sparse_majority_dataset(N_TEST, seed + 10000)
    train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    result = {"seed": seed, "task": "sparse_majority", "n_bits": N_BITS, "k_sparse": K_SPARSE}
    timeline = []
    for step in range(MAX_STEPS):
        model.train()
        idx = torch.randint(0, len(train_x), (BATCH_SIZE,), device=DEVICE)
        logits = model(train_x[idx])
        loss = F.cross_entropy(logits[:, -1], train_y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == MAX_STEPS - 1:
            te_loss, te_acc = evaluate(model, test_x, test_y)
            print(f" step {step:5d} test_loss={te_loss:.4f} acc={te_acc:.3f}", flush=True)
            timeline.append({"step": step, "test_loss": te_loss, "test_acc": te_acc})
            if te_acc > 0.95 and step >= 4000:
                print(" → grokked, stopping early", flush=True)
                break
    result["timeline"] = timeline
    result["final_acc"] = timeline[-1]["test_acc"]
    result["grokked"] = timeline[-1]["test_acc"] > 0.95
    ckpt_path = RESULTS_DIR / f"grokked_majority_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f" Saved → {ckpt_path}", flush=True)
    if not result["grokked"]:
        print(" Not grokked — skipping causal tests", flush=True)
        return result
    print(" Extracting contrastive subspace...", flush=True)
    mem_B = extract_contrastive_subspace(model, train_x, test_x)
    rand_B = torch.linalg.qr(torch.randn(K_DIMS, D_MODEL, device=DEVICE).T, mode="reduced")[0].T
    causal = {}
    for name, B in [("mem", mem_B), ("random", rand_B)]:
        causal[name] = {}
        for coeff in [0.0, 1.0, 2.0]:
            hook = make_cancel_hook(B, coeff) if coeff > 0 else None
            loss, acc = evaluate(model, test_x, test_y, hook=hook)
            causal[name][f"c={coeff}"] = {"loss": loss, "acc": acc}
            print(f" {name:8s} c={coeff:.1f} → loss={loss:.4f} acc={acc:.3f}", flush=True)
    result["causal"] = causal
    loss_c, acc_c = evaluate(model, test_x, test_y, hook=make_cancel_hook(mem_B, 2.0))
    loss_r, acc_r = evaluate(model, test_x, test_y)
    result["recovery"] = {"cancelled_acc": acc_c, "recovered_acc": acc_r}
    print(f" recovery: cancelled={acc_c:.3f} recovered={acc_r:.3f}", flush=True)
    frac = logit_contribution(model, test_x, test_y, mem_B)
    result["logit_frac_mem"] = frac
    print(f" logit fraction (mem) = {frac:.3f}", flush=True)
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
    out = RESULTS_DIR / f"majority_{args.start}_{args.end}_{stamp}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)

if __name__ == "__main__":
    main()
