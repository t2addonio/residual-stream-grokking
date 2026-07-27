import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import random
import numpy as np
import json
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
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
# CONFIGURATION
# ======================
results = []

# Control (no clipping)
for seed in range(100, 115):  # 15 seeds
    res = run_experiment(seed=seed, max_norm=999.0, name="Control")
    results.append(res)

# Static Per-Row 1.8
for seed in range(200, 215):  # 15 seeds
    res = run_experiment(seed=seed, max_norm=1.8, name="Static 1.8")
    results.append(res)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"../results/static_1.8_results_{timestamp}.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
from collections import defaultdict
summary = defaultdict(list)
for r in results:
    summary[r["name"]].append(r["grokked"])

for name, vals in summary.items():
    rate = sum(vals) / len(vals)
    print(f"{name:<15} | {sum(vals)}/{len(vals)} grokked ({rate:.1%})")
