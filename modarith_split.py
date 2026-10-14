"""Held-out train/test split for the mod-p addition clipping scripts.

The original run_static_* / run_mild_envelope / run_weight_tracking scripts sampled
fresh (a, b) pairs from all p^2 pairs every step and called a run "grokked" when the
final *training-batch* loss was < 0.8. With no held-out set that measures fitting,
not generalization. These helpers give a fixed, seeded split; grokking is judged on
held-out test accuracy.
"""
from __future__ import annotations

import torch

GROK_THRESHOLD = 0.95  # held-out test accuracy needed to count as grokked


def make_split(p: int, seed: int, train_frac: float, device):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.zeros_like(a)], dim=1)
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(p * p * train_frac)
    tr, te = perm[:n_train], perm[n_train:]
    return (x[tr].to(device), y[tr].to(device)), (x[te].to(device), y[te].to(device))


def sample_batch(train, batch_size: int, gen: torch.Generator):
    x, y = train
    idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),), generator=gen)
    idx = idx.to(x.device)
    return x[idx], y[idx]


@torch.no_grad()
def evaluate(model, data, chunk: int = 4096):
    x, y = data
    was_training = model.training
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, x.shape[0], chunk):
        logits = model(x[i:i + chunk])[:, -1]
        loss_sum += torch.nn.functional.cross_entropy(logits, y[i:i + chunk], reduction="sum").item()
        correct += (logits.argmax(-1) == y[i:i + chunk]).sum().item()
    model.train(was_training)
    n = x.shape[0]
    return correct / n, loss_sum / n


def add_split_args(parser):
    parser.add_argument("--max_steps", type=int, default=20001)
    parser.add_argument("--train_frac", type=float, default=0.3,
                        help="fraction of the p^2 pairs used for training (rest held out)")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--grok_threshold", type=float, default=GROK_THRESHOLD,
                        help="held-out test accuracy needed to count a run as grokked")
    parser.add_argument("--out_dir", default="results")
    return parser
