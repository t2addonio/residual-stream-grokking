#!/usr/bin/env python3
"""
worker_joint4.py
Simultaneous cancellation of the top-4 residual directions (coarse grid)
Loads existing dirs.npy from previous multi-direction run.
"""

import argparse, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import json
from itertools import product

# ============================================================
MODEL_NAME = "gpt2"
N_CTX = 128
BATCH_SIZE = 16
TOP_K = 4
SCALES = np.linspace(-1.5, 3.0, 5)   # coarse: 5 values
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prev_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()

TEMPLATES = [
    "Is the bracket structure balanced? Sequence: {dyck}",
    "Check if this is valid: {dyck}",
    "Does this sequence have balanced brackets? {dyck}",
    "Bracket check: {dyck}",
    "Is this a valid Dyck sequence? {dyck}",
]

def is_valid_dyck(seq):
    bal = 0
    for t in seq:
        bal += 1 if t == '(' else -1
        if bal < 0: return False
    return bal == 0

def random_dyck(length, rng):
    if length % 2: length -= 1
    s, bal = [], 0
    for i in range(length):
        if bal == 0 or (bal < length - i and rng.rand() < 0.5):
            s.append('('); bal += 1
        else:
            s.append(')'); bal -= 1
    return ''.join(s)

class DyckInTextDataset(Dataset):
    def __init__(self, tokenizer, n_samples=1200, max_dyck_len=16, seed=0):
        rng = np.random.RandomState(seed)
        self.examples = []
        for _ in range(n_samples):
            length = rng.randint(4, max_dyck_len + 1)
            if rng.rand() > 0.4:
                dyck = random_dyck(length, rng); label = 1
            else:
                dyck = ''.join(rng.choice(['(', ')'], size=length))
                while is_valid_dyck(dyck):
                    dyck = ''.join(rng.choice(['(', ')'], size=length))
                label = 0
            text = rng.choice(TEMPLATES).format(dyck=dyck)
            self.examples.append((text, label))
        self.tokenizer = tokenizer
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        text, label = self.examples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=N_CTX,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)}

def make_multi_hook(dirs, scales):
    dirs = [F.normalize(d.float(), dim=0) for d in dirs]
    def hook(resid, hook):
        for d, s in zip(dirs, scales):
            proj = torch.einsum("bpd,d->bp", resid, d)
            resid = resid - s * proj.unsqueeze(-1) * d
        return resid
    return hook

@torch.no_grad()
def measure(model, classifier, loader, device, dirs, scales):
    model.add_hook(f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
                   make_multi_hook(dirs, scales))
    accs, corrs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        _, cache = model.run_with_cache(input_ids,
            names_filter=lambda n: n.endswith("hook_resid_post"))
        resid = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"][:, -1]
        logits = classifier(resid)
        preds = logits.argmax(-1)
        accs.append((preds == labels).float())
        c = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        corrs.append(c)
    model.reset_hooks()
    return torch.cat(accs).mean().item(), torch.cat(corrs).mean().item()

def main():
    args = parse_args()
    device = torch.device(args.device)
    seed = args.seed
    prev_dir = Path(args.prev_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed {seed}] Loading directions from {prev_dir}")
    dirs_np = np.load(prev_dir / "dirs.npy")
    dirs = [torch.tensor(d, device=device) for d in dirs_np]
    print(f"[seed {seed}] Loaded {len(dirs)} directions")

    print(f"[seed {seed}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    d_model = model.cfg.d_model
    classifier = torch.nn.Linear(d_model, 2).to(device)

    held_ds = DyckInTextDataset(tokenizer, 1200, seed=seed+10000)
    held_loader = DataLoader(held_ds, batch_size=BATCH_SIZE)

    top_dirs = dirs[:TOP_K]
    print(f"[seed {seed}] Joint cancellation on top {TOP_K} directions")

    best_acc = 1.0
    best_scales = None
    best_logit = None
    total = len(SCALES) ** TOP_K
    print(f"[seed {seed}] Searching {total} scale combinations...")

    from tqdm import tqdm
    for scales in tqdm(product(SCALES, repeat=TOP_K), total=total, desc="joint4"):
        acc, logit = measure(model, classifier, held_loader, device, top_dirs, scales)
        if acc < best_acc:
            best_acc = acc
            best_scales = [float(s) for s in scales]
            best_logit = float(logit)

    print(f"[seed {seed}] Best joint acc: {best_acc:.4f}")
    print(f"[seed {seed}] Best scales: {best_scales}")
    print(f"[seed {seed}] Correct logit: {best_logit:.3f}")

    meta = {
        "seed": seed,
        "top_k": TOP_K,
        "best_acc": float(best_acc),
        "best_scales": best_scales,
        "best_correct_logit": best_logit,
        "scales_grid": SCALES.tolist(),
    }
    seed_out = out_dir / f"seed_{seed:03d}"
    seed_out.mkdir(exist_ok=True)
    with open(seed_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[seed {seed}] Saved → {seed_out}")

if __name__ == "__main__":
    main()
