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
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
p = 113

def get_model():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, d_head=32, n_heads=4, d_mlp=512,
        n_ctx=3, d_vocab=p+1, act_fn="relu", positional_embedding_type="standard",
    )
    return HookedTransformer(cfg).to(device)

def make_data(batch_size=512):
    a = torch.randint(0, p, (batch_size, 1), device=device)
    b = torch.randint(0, p, (batch_size, 1), device=device)
    x = torch.cat([a, b, torch.zeros_like(a)], dim=1)
    y = (a + b) % p
    return x, y

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

def run_experiment(seed, max_norm, max_steps=20001, name="Run"):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {name} | Seed {seed} | Max Norm = {max_norm} ===")

    for step in tqdm(range(max_steps)):
        x, y = make_data()
        logits = model(x)
        loss = criterion(logits[:, -1], y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if max_norm < 999:
            apply_per_row_clipping(model, max_norm=max_norm)

        if step % 2000 == 0:
            print(f"Step {step:5d}: Loss = {loss.item():.4f}")

    final_loss = loss.item()
    grokked = final_loss < 0.8
    print(f"Final loss = {final_loss:.4f} | Grokked: {grokked}")
    
    return {
        "name": name,
        "seed": seed,
        "max_norm": max_norm,
        "final_loss": final_loss,
        "grokked": grokked
    }

# ======================
# ARGUMENT PARSING
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=100, help="Start seed")
parser.add_argument("--end", type=int, default=114, help="End seed (inclusive)")
parser.add_argument("--norm", type=float, default=999.0, help="Max row norm (999 = Control)")
parser.add_argument("--name", type=str, default="Control", help="Condition name")
args = parser.parse_args()

# Create results directory
os.makedirs("results", exist_ok=True)

results = []

for seed in range(args.start, args.end + 1):
    res = run_experiment(seed=seed, max_norm=args.norm, name=args.name)
    results.append(res)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/{args.name.replace(' ', '_')}_{args.start}_{args.end}_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {filename}")

# Summary
print("\n" + "="*60)
print(f"SUMMARY: {args.name}")
print("="*60)
successes = sum(1 for r in results if r["grokked"])
print(f"{successes}/{len(results)} grokked ({successes/len(results):.1%})")
