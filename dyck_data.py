#!/usr/bin/env python3
"""
dyck_data.py

Generate Dyck language (balanced parentheses) datasets with
controlled depth / length and clean train vs held-out splits.
"""

import torch
import random
from typing import List, Tuple

OPEN = 0
CLOSE = 1
VOCAB = {0: "(", 1: ")"}

def is_balanced(seq: List[int]) -> bool:
    depth = 0
    for t in seq:
        depth += 1 if t == OPEN else -1
        if depth < 0:
            return False
    return depth == 0

def generate_balanced(max_len: int, max_depth: int) -> List[int]:
    """Generate one balanced sequence with length <= max_len and depth <= max_depth."""
    while True:
        seq = []
        depth = 0
        for _ in range(max_len):
            # Prefer open if we still have budget, prefer close if deep
            if depth == 0:
                choice = OPEN
            elif depth >= max_depth or len(seq) > max_len - depth - 1:
                choice = CLOSE
            else:
                choice = random.choice([OPEN, CLOSE])
            seq.append(choice)
            depth += 1 if choice == OPEN else -1
            if depth < 0:
                break
        if depth == 0 and 2 <= len(seq) <= max_len:
            return seq

def generate_unbalanced(max_len: int, max_depth: int) -> List[int]:
    """Generate a clearly unbalanced sequence."""
    while True:
        seq = [random.choice([OPEN, CLOSE]) for _ in range(random.randint(2, max_len))]
        if not is_balanced(seq):
            return seq

def make_dataset(
    n_samples: int,
    max_len: int = 20,
    max_depth: int = 4,
    balanced_ratio: float = 0.5,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x: [n_samples, max_len]  (padded with a special pad token = 2)
        y: [n_samples]           (1 = balanced, 0 = unbalanced)
    """
    random.seed(seed)
    PAD = 2
    xs, ys = [], []
    n_bal = int(n_samples * balanced_ratio)

    for i in range(n_samples):
        if i < n_bal:
            seq = generate_balanced(max_len, max_depth)
            label = 1
        else:
            seq = generate_unbalanced(max_len, max_depth)
            label = 0
        # pad
        padded = seq + [PAD] * (max_len - len(seq))
        xs.append(padded)
        ys.append(label)

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

def train_heldout_split(
    n_train: int = 8000,
    n_heldout: int = 2000,
    max_len_train: int = 16,
    max_depth_train: int = 3,
    max_len_heldout: int = 24,
    max_depth_heldout: int = 5,
    seed: int = 42,
):
    """
    Train: shorter / shallower sequences
    Held-out: longer / deeper sequences  (generalization test)
    """
    x_train, y_train = make_dataset(
        n_train, max_len=max_len_train, max_depth=max_depth_train, seed=seed
    )
    x_held, y_held = make_dataset(
        n_heldout, max_len=max_len_heldout, max_depth=max_depth_heldout, seed=seed + 1
    )
    return (x_train, y_train), (x_held, y_held)

if __name__ == "__main__":
    (x_tr, y_tr), (x_ho, y_ho) = train_heldout_split()
    print("Train:", x_tr.shape, y_tr.float().mean().item())
    print("Held-out:", x_ho.shape, y_ho.float().mean().item())
