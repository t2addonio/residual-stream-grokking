import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import random
import numpy as np
import json
import os
import argparse
from datetime import datetime
from modarith_split import make_split, sample_batch, evaluate, add_split_args
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
p = 113

def get_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, d_head=32, n_heads=4, d_mlp=512,
        n_ctx=3, d_vocab=p+1, act_fn="relu", positional_embedding_type="standard",
    )
    return HookedTransformer(cfg).to(device)

def apply_per_row_clipping(model, max_norm=1.8):
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                original_shape = param.shape
                flat = param.view(original_shape[0], -1)
                norms = flat.norm(dim=1, keepdim=True)
                scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
                flat.mul_(scale)
                param.copy_(flat.view(original_shape))

def run_experiment(seed, max_norm, max_steps=20001, name="Run", train_frac=0.3, eval_every=500, grok_threshold=0.95):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    gen = torch.Generator().manual_seed(seed)
    train, test = make_split(p, seed, train_frac, device)
    history = []  # (step, train_acc, test_acc, test_loss)
    memorize_step = grok_step = None
    print(f"\n=== {name} | Seed {seed} | Max Norm = {max_norm} ===")

    for step in tqdm(range(max_steps)):
        x, y = sample_batch(train, 512, gen)
        logits = model(x)
        loss = criterion(logits[:, -1], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if max_norm < 999:
            apply_per_row_clipping(model, max_norm=max_norm)

        if step % eval_every == 0 or step == max_steps - 1:
            tr_acc, _ = evaluate(model, train)
            te_acc, te_loss = evaluate(model, test)
            history.append((step, tr_acc, te_acc, te_loss))
            if memorize_step is None and tr_acc >= 0.99:
                memorize_step = step
            if grok_step is None and te_acc >= grok_threshold:
                grok_step = step

        if step % 2000 == 0:
            print(f"Step {step:5d}: Loss = {loss.item():.4f}")

    final_loss = loss.item()
    # Grokking is judged on held-out test accuracy (was: final train-batch loss < 0.8).
    grokked = history[-1][2] >= grok_threshold
    print(f"Final train loss = {final_loss:.4f} | train acc = {history[-1][1]:.3f} | "
          f"test acc = {history[-1][2]:.3f} | grok_step = {grok_step} | Grokked: {grokked}")
    
    return {
        "name": name,
        "seed": seed,
        "max_norm": max_norm,
        "final_loss": final_loss,
        "grokked": grokked,
        "train_acc": history[-1][1],
        "test_acc": history[-1][2],
        "test_loss": history[-1][3],
        "memorize_step": memorize_step,
        "grok_step": grok_step,
        "train_frac": train_frac,
        "grok_threshold": grok_threshold,
        "eval_history": history
    }

# ======================
# CONFIGURATION
# ======================
CONDITIONS = [  # (name, first seed, kwargs)
    ('Control', 100, dict(max_norm=999.0)),
    ('Static 1.8', 200, dict(max_norm=1.8)),
]

parser = argparse.ArgumentParser()
parser.add_argument("--seeds_per_condition", type=int, default=15)
parser.add_argument("--only", nargs="*", default=None, help="run only these condition names")
add_split_args(parser)
args = parser.parse_args()

def _summary(results, title):
    print("\n" + "=" * 70)
    print(title + "  (grokked = held-out test acc >= %.2f)" % args.grok_threshold)
    print("=" * 70)
    groups = defaultdict(list)
    for r in results:
        groups[r["name"]].append(r)
    for name, rs in groups.items():
        g = sum(r["grokked"] for r in rs)
        mean_te = sum(r["test_acc"] for r in rs) / len(rs)
        print(f"{name:<20} | {g}/{len(rs)} grokked ({g/len(rs):.1%}) | mean test acc {mean_te:.3f}")

os.makedirs(args.out_dir, exist_ok=True)
results = []
for name, seed0, kw in CONDITIONS:
    if args.only and name not in args.only:
        continue
    for seed in range(seed0, seed0 + args.seeds_per_condition):
        results.append(run_experiment(seed=seed, name=name, max_steps=args.max_steps, train_frac=args.train_frac, eval_every=args.eval_every, grok_threshold=args.grok_threshold, **kw))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(args.out_dir, f"static_1.8_results_{timestamp}.json")
with open(filename, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {filename}")
_summary(results, 'SUMMARY')
