#!/usr/bin/env python3
"""
analyze_directions.py

Tier-1 mechanistic analysis of the contrastive residual subspace.

For each grokked checkpoint:
  1. Extract 16-D contrastive subspace (same method as training runs)
  2. Rank each dimension by causal damage when ablated alone
  3. List max-activating examples per dimension
  4. List examples most damaged by cancelling that dimension
  5. Linear probes: does each dim / the full subspace encode
     sparse bits, label, or margin?

Usage:
  python analyze_directions.py --task parity --ckpt_dir results_sparse_parity --seeds 30001 30004 30010
  python analyze_directions.py --task majority --ckpt_dir results_sparse_majority --seeds 31000 31001 31002
  python analyze_directions.py --task parity --ckpt_dir results_sparse_parity --seeds 30001 --topk 15
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

N_BITS = 20
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 64
D_MLP = 1024
K_DIMS = 16
LAYER = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_CFG = {
    "parity": {"k_sparse": 4, "ckpt_prefix": "grokked_parity_seed", "label_fn": "parity"},
    "majority": {"k_sparse": 5, "ckpt_prefix": "grokked_majority_seed", "label_fn": "majority"},
}


def make_model():
    cfg = HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        act_fn="relu",
        normalization_type=None,
        d_vocab=2,
        n_ctx=N_BITS,
        device=DEVICE,
    )
    return HookedTransformer(cfg)


def make_dataset(task, n_samples, seed):
    k = TASK_CFG[task]["k_sparse"]
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n_samples, N_BITS), generator=g)
    if task == "parity":
        y = x[:, :k].sum(dim=1) % 2
    else:
        y = (x[:, :k].sum(dim=1) > (k // 2)).long()
    return x, y


def get_residuals(model, x, layer=LAYER):
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(x.to(DEVICE))
        return cache[f"blocks.{layer}.hook_resid_post"].mean(dim=1).detach()


def extract_contrastive_subspace(model, train_x, test_x, k=K_DIMS):
    r_tr = get_residuals(model, train_x)
    r_te = get_residuals(model, test_x)
    mu_tr = r_tr.mean(dim=0)
    mu_te = r_te.mean(dim=0)
    contrast = mu_tr - mu_te
    centered = r_tr - mu_tr
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    B = Vh[:k].clone()
    c = contrast / (contrast.norm() + 1e-8)
    B[0] = c
    B = torch.linalg.qr(B.T, mode="reduced")[0].T
    return B


def project(r, B):
    return r @ B.T


def make_cancel_hook_dims(B, dim_indices, coeff=2.0):
    if len(dim_indices) == 0:
        return None
    B_sub = B[dim_indices]

    def hook(act, hook):
        flat = act.reshape(-1, act.shape[-1])
        coeffs = flat @ B_sub.T
        flat = flat - coeff * (coeffs @ B_sub)
        return flat.reshape(act.shape)

    return hook


def evaluate(model, x, y, hook=None):
    model.eval()
    with torch.no_grad():
        handle = model.blocks[LAYER].hook_resid_post.add_hook(hook) if hook is not None else None
        logits = model(x.to(DEVICE))[:, -1, :]
        if handle is not None:
            handle.remove()
        loss = F.cross_entropy(logits, y.to(DEVICE)).item()
        pred = logits.argmax(dim=-1)
        acc = (pred == y.to(DEVICE)).float().mean().item()
        return loss, acc, logits.detach()


def correct_logit(logits, y):
    return logits[torch.arange(len(y), device=logits.device), y.to(logits.device)]


def analyze_seed(task, seed, ckpt_path, topk=12, n_train=8000, n_test=3000):
    print(f"\n===== {task} seed {seed} =====", flush=True)
    result = {"task": task, "seed": seed, "checkpoint": str(ckpt_path)}

    model = make_model().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("  model loaded", flush=True)

    train_x, train_y = make_dataset(task, n_train, seed=seed)
    test_x, test_y = make_dataset(task, n_test, seed=seed + 10_000)

    loss0, acc0, logits0 = evaluate(model, test_x, test_y)
    result["baseline"] = {"loss": loss0, "acc": acc0}
    print(f"  baseline acc={acc0:.3f}", flush=True)

    B = extract_contrastive_subspace(model, train_x, test_x, k=K_DIMS)
    result["subspace_norms"] = B.norm(dim=1).tolist()

    hook_all = make_cancel_hook_dims(B, list(range(K_DIMS)), coeff=2.0)
    loss_a, acc_a, _ = evaluate(model, test_x, test_y, hook=hook_all)
    result["full_cancel"] = {"loss": loss_a, "acc": acc_a}
    print(f"  full mem cancel acc={acc_a:.3f}", flush=True)

    dim_rank = []
    r_test = get_residuals(model, test_x)
    coeffs_test = project(r_test, B)
    base_correct = correct_logit(logits0, test_y).cpu()

    for d in range(K_DIMS):
        hook = make_cancel_hook_dims(B, [d], coeff=2.0)
        loss_d, acc_d, logits_d = evaluate(model, test_x, test_y, hook=hook)
        correct_d = correct_logit(logits_d, test_y).cpu()
        delta_logit = (base_correct - correct_d).mean().item()
        dim_rank.append({
            "dim": d,
            "acc": acc_d,
            "loss": loss_d,
            "mean_delta_correct_logit": delta_logit,
            "mean_abs_coeff": coeffs_test[:, d].abs().mean().item(),
        })
        print(f"  dim {d:2d}: acc={acc_d:.3f}  Δlogit={delta_logit:.3f}", flush=True)

    dim_rank.sort(key=lambda z: z["acc"])
    result["single_dim_rank"] = dim_rank

    max_act = {}
    x_cpu = test_x.cpu()
    y_cpu = test_y.cpu()
    for d in range(K_DIMS):
        vals = coeffs_test[:, d]
        top_idx = vals.abs().argsort(descending=True)[:topk]
        examples = []
        for i in top_idx.tolist():
            examples.append({
                "idx": i,
                "coeff": vals[i].item(),
                "bits": x_cpu[i].tolist(),
                "label": int(y_cpu[i]),
                "sparse_bits": x_cpu[i, :TASK_CFG[task]["k_sparse"]].tolist(),
            })
        max_act[f"dim_{d}"] = examples
    result["max_activating"] = max_act

    max_dmg = {}
    for d in range(K_DIMS):
        hook = make_cancel_hook_dims(B, [d], coeff=2.0)
        _, _, logits_d = evaluate(model, test_x, test_y, hook=hook)
        correct_d = correct_logit(logits_d, test_y).cpu()
        drop = base_correct - correct_d
        top_idx = drop.argsort(descending=True)[:topk]
        examples = []
        for i in top_idx.tolist():
            examples.append({
                "idx": i,
                "logit_drop": drop[i].item(),
                "bits": x_cpu[i].tolist(),
                "label": int(y_cpu[i]),
                "sparse_bits": x_cpu[i, :TASK_CFG[task]["k_sparse"]].tolist(),
                "coeff": coeffs_test[i, d].item(),
            })
        max_dmg[f"dim_{d}"] = examples
    result["max_damaging"] = max_dmg

    r_tr = get_residuals(model, train_x)
    coeffs_tr = project(r_tr, B)
    k_sp = TASK_CFG[task]["k_sparse"]

    probes = {}
    target_tr, target_te = train_y.float(), test_y.float()
    X = coeffs_tr.cpu()
    XtX = X.T @ X + 1e-3 * torch.eye(K_DIMS)
    w = torch.linalg.solve(XtX, X.T @ target_tr)
    pred = coeffs_test.cpu() @ w
    pred_bin = (pred > 0.5).float()
    acc_probe = (pred_bin == target_te).float().mean().item()
    probes["label"] = {"acc": acc_probe, "corr": torch.corrcoef(torch.stack([pred, target_te]))[0, 1].item()}

    bit_accs = []
    for b in range(k_sp):
        target_tr = train_x[:, b].float()
        target_te = test_x[:, b].float()
        X = coeffs_tr.cpu()
        XtX = X.T @ X + 1e-3 * torch.eye(K_DIMS)
        w = torch.linalg.solve(XtX, X.T @ target_tr)
        pred = coeffs_test.cpu() @ w
        pred_bin = (pred > 0.5).float()
        bit_accs.append((pred_bin == target_te).float().mean().item())
    probes["sparse_bits"] = bit_accs
    probes["sparse_bits_mean"] = sum(bit_accs) / len(bit_accs)

    rand_B = torch.linalg.qr(torch.randn(K_DIMS, D_MODEL, device=DEVICE).T, mode="reduced")[0].T
    coeffs_tr_r = project(r_tr, rand_B)
    coeffs_te_r = project(r_test, rand_B)
    X = coeffs_tr_r.cpu()
    XtX = X.T @ X + 1e-3 * torch.eye(K_DIMS)
    w = torch.linalg.solve(XtX, X.T @ train_y.float())
    pred = coeffs_te_r.cpu() @ w
    pred_bin = (pred > 0.5).float()
    probes["random_subspace_label_acc"] = (pred_bin == test_y.float()).float().mean().item()

    result["probes"] = probes
    print(f"  probe label acc={probes['label']['acc']:.3f}  bits mean={probes['sparse_bits_mean']:.3f}  random ctrl={probes['random_subspace_label_acc']:.3f}", flush=True)

    top_dims = [z["dim"] for z in dim_rank[:5]]
    result["top5_damaging_dims"] = top_dims
    print(f"  top-5 damaging dims: {top_dims}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["parity", "majority"])
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default="results_direction_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    prefix = TASK_CFG[args.task]["ckpt_prefix"]

    all_results = []
    for seed in args.seeds:
        candidates = [
            ckpt_dir / f"{prefix}{seed}.pt",
            ckpt_dir / f"grokked_{args.task}_seed{seed}.pt",
            Path(f"{prefix}{seed}.pt"),
            Path(f"results_sparse_{args.task}") / f"{prefix}{seed}.pt",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            print(f"Checkpoint not found for seed {seed} — tried {candidates[0]}", flush=True)
            continue
        all_results.append(analyze_seed(args.task, seed, ckpt, topk=args.topk))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dirs_{args.task}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
