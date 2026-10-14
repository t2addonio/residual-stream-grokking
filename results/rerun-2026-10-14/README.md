# CPU re-runs with the fixed code (2026-10-14)

Re-runs on `main@6848a00` (after PR #2: real test split for the clipping scripts, Dyck accuracy/pad-token fix, shuffled held-out set, CPU fallback). CPU only, torch 2.14.1, transformer-lens 3.9.0.

- [`RERUN_SUMMARY.md`](RERUN_SUMMARY.md): full write-up. Absolute paths in it (`/workspace/research/runs/...`) refer to the machine the runs were made on; the matching files are in this folder.
- `dyck30/`: Dyck 30-seed (seeds 100–129) causal results + direction analysis. `summary.txt` = new; `old_committed_summary.txt` = the superseded table re-classified. Checkpoints (`*.pt`) are not committed.
- `v5-random-control/`: v5 random control, c=2, seeds 0–4, d128/d256, addition and multiplication (JSON only).
- `clipping/`: 27 clipping runs (9 conditions × 3 seeds), per-run JSON with eval curves.
- `plots/`: `dyck30_cancel.png`, `v5_random_control.png`, `clipping_test_acc.png`.
- `make_plots.py`: aggregation/plot script (expects the original folder names `2026-10-14-*`; adjust paths to run here).

Headlines:

| Experiment | Result |
|---|---|
| Dyck 30 seeds, contrastive cancel c=2 | 30/30 label flips, acc 0.007 ± 0.014 (base 0.993 ± 0.013); controls intact. Contrastive subspace aligns 0.977 ± 0.014 with the readout direction, so the flip is largely built in. Algorithmic subspace is rank 3, not 8. Overlap metric > 1 (normalisation bug): old overlap values withdrawn. |
| v5 random control, c=2 (5 seeds) | add d128 0.403 ± 0.175 · mul d128 0.423 ± 0.204 · add d256 0.985 ± 0.017 · mul d256 0.947 ± 0.086. Contrastive → 0.000 everywhere. |
| Clipping (27 runs) | unclipped controls 6/9 end ≥ 0.95 (mean 0.90); clipped/envelope 7/18 (mean 0.68). No grokking delay; all conditions unstable. |

See [`../../ERRATA.md`](../../ERRATA.md) for what this changes in the papers.
