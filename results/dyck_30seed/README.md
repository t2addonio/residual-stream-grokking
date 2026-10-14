# Dyck 30-seed contrastive phase-cancellation

From-scratch 2-layer `d_model=128` Dyck models, seeds 100–129.
Intervention: final residual stream (`blocks.1.hook_resid_post`), coeff=2.

> **SUPERSEDED (2026-10-14).** The table below was produced by a version of `train_dyck_30seed.py` whose accuracy counted the pad token as a class (and whose early-stop slice was unshuffled). The "9/30 strong collapse / 18 near chance" split is an artefact of that bug. Re-run with the fixed code: [`../rerun-2026-10-14/`](../rerun-2026-10-14/). Kept here unchanged for the record.

## Corrected headlines (re-run 2026-10-14, fixed code, seeds 100–129)

- Base held-out accuracy: 0.993 ± 0.013 (seed 118: 0.924)
- Contrastive cancel (c=2): **30/30 seeds label-flip**, accuracy 0.007 ± 0.014 (max 0.078); top-1 direction alone 0.033 ± 0.099
- Algorithmic control 0.993 ± 0.013; random control 0.992 ± 0.019: intact on all seeds
- **Caveat:** the contrastive subspace aligns 0.977 ± 0.014 with the readout logit-difference direction (algorithmic 0.18, random 0.24). A label flip at c=2 is therefore largely expected by construction and is not, by itself, evidence of a separate component. A readout-matched control is needed.
- The algorithmic control subspace is **rank 3, not 8**; on 23/30 seeds its cancel equals base exactly (hook close to a no-op).
- The `overlap` column below is **invalid**: the metric's normalisation is wrong (re-run values 2.73 ± 0.49, i.e. > 1).

## Original headlines (superseded)

- Base held-out accuracy: mean 0.994
- Strong collapse (cancel acc ≤ 0.03): **9/30** seeds [100, 110, 115, 116, 117, 118, 127, 128, 129]
- Moderate drop: 3 seeds [101, 103, 126]
- Chance / near-chance (≥ 0.4): 18 seeds
- Algorithmic (stack/count) and random controls: no collapse on any seed

## Per-seed table (superseded, original run)

| seed | base | cancel contrastive | cancel algo | cancel random | overlap |
|------|------|--------------------|-------------|---------------|---------|
| 100 | 0.992 | 0.015 | 0.992 | 0.992 | 0.0134 |
| 101 | 0.992 | 0.180 | 0.992 | 0.992 | 0.0070 |
| 102 | 0.989 | 0.500 | 0.989 | 0.989 | 0.0031 |
| 103 | 0.992 | 0.031 | 0.992 | 0.992 | 0.0091 |
| 104 | 0.999 | 0.500 | 0.999 | 0.999 | 0.0003 |
| 105 | 0.994 | 0.500 | 0.994 | 0.994 | 0.0508 |
| 106 | 0.995 | 0.500 | 0.995 | 0.995 | 0.0005 |
| 107 | 0.996 | 0.500 | 0.996 | 0.996 | 0.0023 |
| 108 | 1.000 | 0.500 | 1.000 | 1.000 | 0.0008 |
| 109 | 0.992 | 0.485 | 0.992 | 0.992 | 0.0018 |
| 110 | 0.992 | 0.009 | 0.992 | 0.992 | 0.0006 |
| 111 | 0.988 | 0.485 | 0.988 | 0.988 | 0.0020 |
| 112 | 0.990 | 0.500 | 0.990 | 0.990 | 0.0025 |
| 113 | 0.992 | 0.500 | 0.992 | 0.992 | 0.0299 |
| 114 | 0.996 | 0.500 | 0.996 | 0.996 | 0.0014 |
| 115 | 0.994 | 0.009 | 0.994 | 0.994 | 0.0171 |
| 116 | 0.997 | 0.005 | 0.997 | 0.997 | 0.0014 |
| 117 | 0.996 | 0.005 | 0.996 | 0.996 | 0.0037 |
| 118 | 0.997 | 0.005 | 0.997 | 0.997 | 0.0050 |
| 119 | 1.000 | 0.419 | 1.000 | 1.000 | 0.0001 |
| 120 | 0.996 | 0.500 | 0.996 | 0.996 | 0.0004 |
| 121 | 0.993 | 0.500 | 0.993 | 0.993 | 0.0078 |
| 122 | 0.997 | 0.500 | 0.997 | 0.997 | 0.0079 |
| 123 | 0.997 | 0.500 | 0.997 | 0.997 | 0.0002 |
| 124 | 0.992 | 0.500 | 0.992 | 0.992 | 0.0006 |
| 125 | 0.996 | 0.500 | 0.996 | 0.996 | 0.0052 |
| 126 | 0.998 | 0.243 | 0.998 | 0.998 | 0.0019 |
| 127 | 0.991 | 0.006 | 0.991 | 0.998 | 0.0140 |
| 128 | 0.995 | 0.015 | 0.995 | 0.995 | 0.0237 |
| 129 | 0.996 | 0.006 | 0.996 | 0.996 | 0.0048 |

## Direction analysis (strong-collapse seeds; original run, not re-checked)

When collapse is strong, top-1 cancellation is already sufficient.
Max-activating sequences are short illegal prefixes (`)(`, `))`, `)))`).

Raw JSON: `dyck_30seed_causal.json`, `dyck_30seed_direction_analysis.json`.
Checkpoints (`seed_*.pt`, ~1.6MB each) are not in git.
