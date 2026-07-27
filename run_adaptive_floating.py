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

def get_per_row_norms(model):
    norms_list = []
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                flat = param.view(param.shape[0], -1)
                norms = flat.norm(dim=1)
                norms_list.append(norms.clone())
    return norms_list

def apply_adaptive_clipping(model, base_norm, row_scales, global_floor=1.2):
    with torch.no_grad():
        param_idx = 0
        for param in model.parameters():
            if param.ndim >= 2:
                original_shape = param.shape
                flat = param.view(original_shape[0], -1)
                norms = flat.norm(dim=1, keepdim=True)
                max_norms = torch.clamp(base_norm * row_scales[param_idx], min=global_floor).unsqueeze(1)
                scale = torch.clamp(max_norms / (norms + 1e-8), max=1.0)
                flat.mul_(scale)
                param.copy_(flat.view(original_shape))
                param_idx += 1

def run_adaptive(
    seed,
    max_steps=20001,
    base_norm=2.0,
    measure_every=100,
    strength=0.65,
    global_floor=1.2,
    name="Adaptive"
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {name} | Seed {seed} | Base Norm {base_norm} | Strength {strength} ===")

    prev_norms = get_per_row_norms(model)
    row_scales = [torch.ones_like(n) for n in prev_norms]

    for step in tqdm(range(max_steps)):
        x, y = make_data()
        logits = model(x)
        loss = criterion(logits[:, -1], y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % measure_every == 0 and step > 0:
            current_norms = get_per_row_norms(model)
            new_scales = []
            for prev, curr in zip(prev_norms, current_norms):
                growth = (curr - prev) / measure_every
                median_growth = growth.median()
                relative = growth - median_growth
                relative = relative / (relative.abs().median() + 1e-6)
                scale = torch.ones_like(growth)
                high_growth_mask = relative > 0.5
                scale[high_growth_mask] = 1.0 - strength * torch.clamp(relative[high_growth_mask], max=1.5)
                scale = torch.clamp(scale, min=0.4, max=1.0)
                new_scales.append(scale)
            row_scales = new_scales
            prev_norms = current_norms

        apply_adaptive_clipping(model, base_norm, row_scales, global_floor=global_floor)

        if step % 2000 == 0:
            print(f"Step {step:5d}: Loss={loss.item():.4f}")

    final_loss = loss.item()
    grokked = final_loss < 0.8
    print(f"Final loss = {final_loss:.4f} | Grokked: {grokked}")

    return {
        "name": name,
        "seed": seed,
        "final_loss": final_loss,
        "grokked": grokked,
        "base_norm": base_norm,
        "strength": strength
    }

# ======================
# ARGUMENT PARSING
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True, help="Start seed")
parser.add_argument("--end", type=int, required=True, help="End seed (inclusive)")
parser.add_argument("--norm", type=float, default=2.0, help="Base max norm")
parser.add_argument("--strength", type=float, default=0.65, help="Adaptation strength (0.0 = static)")
parser.add_argument("--name", type=str, default="Adaptive", help="Condition name")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

results = []
for seed in range(args.start, args.end + 1):
    res = run_adaptive(
        seed=seed,
        base_norm=args.norm,
        strength=args.strength,
        name=args.name
    )
    results.append(res)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/{args.name.replace(' ', '_')}_{args.start}_{args.end}_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {filename}")

# Summary
successes = sum(1 for r in results if r["grokked"])
print(f"\n{args.name}: {successes}/{len(results)} grokked ({successes/len(results):.1%})")
