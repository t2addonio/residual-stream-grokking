# Results

Per-seed tables extracted from the working-store archives on 2026-09-05.

- [`SUMMARY.md`](SUMMARY.md) — headline numbers + full seed tables (dual-80, 256, 512×4, GPT-2)
- [`rerun-2026-10-14/`](rerun-2026-10-14/) — **CPU re-runs with fixed code** (Dyck 30-seed, v5 random control, clipping)
- [`dyck_30seed/README.md`](dyck_30seed/README.md) — from-scratch Dyck 30-seed contrastive phase-cancellation (seeds 100–129)
- JSON dumps of later `meta.json` / `sequential.json` / `pairs.json` live in the working pack and can be regenerated from the tars

## Headlines

| Experiment | n | collapse (`min_acc` < 0.05) | mean min_acc | mean final_acc |
|------------|---|------------------------------|--------------|----------------|
| Dual interference, pure Dyck | 80 | 80/80 (100%) | 0.0016 | 0.9977 |
| Wider Dyck d_model=256 | 40 | 40/40 (100%) | 0.0018 | 0.9979 |
| Deep-wide Dyck 512×4 | 40 | 40/40 (100%) | 0.0019 | 0.9972 |
| GPT-2 Dyck-in-Text dual | 24 | 0/24 (0%) | 0.3976 | 0.5963 |
| GPT-2 joint-4 | 8 | 0/8 (0%) | 0.3952 (best_acc) | — |
| ~~Contrastive cancel, Dyck 128×2~~ (superseded) | 30 | ~~9/30 strong (≤0.03); 18/30 ~chance~~ (metric bug) | — | 0.994 |
| Contrastive cancel, Dyck 128×2, **re-run 2026-10-14** | 30 | 30/30 label flip (acc 0.007 ± 0.014); readout-aligned 0.977 | 0.007 | 0.993 |

~~Interpretation: concentrated / low-rank residual support on from-scratch Dyck; distributed support on weakly fine-tuned GPT-2. The earlier contrastive-extraction protocol is seed-variable on Dyck 128×2 (9/30 strong collapse); later dual-interference recovers collapse on 80/80.~~

**Corrected interpretation (2026-10-14):** the "seed-variable 9/30" result was a metric bug; with the fix the contrastive cancel flips labels on 30/30 seeds, but that subspace is ~98% aligned with the readout, so the flip is largely built in. "Concentrated vs distributed" is a hypothesis: the GPT-2 model is near chance (0.55–0.62). See [`../ERRATA.md`](../ERRATA.md).
