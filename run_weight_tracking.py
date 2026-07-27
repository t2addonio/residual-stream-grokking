import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm import tqdm
import random
import numpy as np
import json
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

def get_average_row_norm(model):
    total_norm = 0.0
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                flat = param.view(param.shape[0], -1)
                norms = flat.norm(dim=1)
                total_norm += norms.sum().item()
                count += norms.numel()
    return total_norm / count if count > 0 else 0.0

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

def run_with_weight_tracking(
    seed,
    max_steps=20001,
    base_norm=1.8,
    use_envelope=False,
    peak_start=300,
    peak_end=1200,
    min_norm=1.4,
    name="Run"
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {name} | Seed {seed} | Base Norm {base_norm} | Envelope={use_envelope} ===")

    steps_logged = []
    losses_logged = []
    avg_row_norms = []
    applied_norms = []

    for step in tqdm(range(max_steps)):
        x, y = make_data()
        logits = model(x)
        loss = criterion(logits[:, -1], y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Determine current max_norm
        if use_envelope:
            if step < peak_start:
                current_norm = base_norm
            elif step <= peak_end:
                progress = (step - peak_start) / (peak_end - peak_start)
                current_norm = base_norm - progress * (base_norm - min_norm)
            else:
                current_norm = min_norm
        else:
            current_norm = base_norm

        if current_norm < 999:
            apply_per_row_clipping(model, max_norm=current_norm)

        # Log every 1000 steps
        if step % 1000 == 0 or step == max_steps - 1:
            avg_norm = get_average_row_norm(model)
            steps_logged.append(step)
            losses_logged.append(loss.item())
            avg_row_norms.append(avg_norm)
            applied_norms.append(current_norm)

            if step % 2000 == 0:
                print(f"Step {step:5d}: Loss={loss.item():.4f} | Applied={current_norm:.2f} | Actual Avg={avg_norm:.3f}")

    final_loss = loss.item()
    grokked = final_loss < 0.8
    print(f"Final loss = {final_loss:.4f} | Grokked: {grokked}")

    return {
        "name": name,
        "seed": seed,
        "final_loss": final_loss,
        "grokked": grokked,
        "steps": steps_logged,
        "losses": losses_logged,
        "avg_row_norms": avg_row_norms,
        "applied_norms": applied_norms,
        "use_envelope": use_envelope,
        "base_norm": base_norm
    }

# ======================
# CONFIGURATION - EDIT THIS SECTION FOR EACH GPU
# ======================

results = []

# Example for one GPU (you can change the seed ranges on each machine)

# Control
for seed in range(3000, 3015):  # 15 seeds
    results.append(run_with_weight_tracking(
        seed=seed, base_norm=999.0, use_envelope=False, name="Control"
    ))

# Static 1.8
for seed in range(3100, 3115):  # 15 seeds
    results.append(run_with_weight_tracking(
        seed=seed, base_norm=1.8, use_envelope=False, name="Static 1.8"
    ))

# Weight Envelope
for seed in range(3200, 3215):  # 15 seeds
    results.append(run_with_weight_tracking(
        seed=seed,
        base_norm=1.8,
        use_envelope=True,
        peak_start=300,
        peak_end=1200,
        min_norm=1.4,
        name="Weight Envelope"
    ))

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"weight_tracking_results_{timestamp}.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "="*70)
print("WEIGHT TRACKING SUMMARY")
print("="*70)

summary = defaultdict(list)
for r in results:
    summary[r["name"]].append(r["grokked"])

for name, vals in summary.items():
    rate = sum(vals) / len(vals)
    print(f"{name:<20} | {sum(vals)}/{len(vals)} grokked ({rate:.1%})")
