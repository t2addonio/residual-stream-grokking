# Index — residual-stream-grokking

Catalog of this repository and of companion working-store artifacts (2026-09-05, updated 2026-09-08).

## Companion repositories

- https://github.com/t2addonio/residual-stream-grokking — this repo
- https://github.com/t2addonio/isolate-rescue-grokking — isolate / rescue / freeze science pack
- https://github.com/t2addonio/residual-causal-toolkit — cross-domain residual toolkit
- https://github.com/t2addonio/hybrid-lab — local multi-agent lab

## In this repo (git)

### Paper
- `Residual_Stream_Component_Grokking.pdf` — v5 PDF
- `papers/Residual_Stream_Component_Grokking_v6.md` — **v6 paper text**
- `papers/README.md` — version map + binary upload recipe
- `papers/FIGURES.md` — Figures 5–9 captions
- `results/SUMMARY.md` — per-seed tables (dual-80, 256, 512×4, GPT-2 dual, joint-4)
- `results/README.md` — headline table
- `results/dyck_30seed/` — contrastive phase-cancellation, seeds 100–129 (+ early 42–51)

### Joint-4 transfer experiment
- `experiments/joint4/run_joint4.py`
- `experiments/joint4/worker_joint4.py`
- `experiments/joint4/README.md`

### Original discovery / causal scripts (root)
- `train_dyck.py`, `train_dyck_30seed.py`, `train_dyck_multiseed.py`, `dyck_data.py`
- `analyze_dyck_directions.py`
- `run_discovery_*.py`
- `causal_eval_mult_512.py`
- `fourier_overlap_test.py`, `positive_control_fourier.py`, `positive_control_p257.py`
- `group_ablation.py`, `layer_sweep.py`, `orthogonal_complement_sweep.py`
- `logit_contribution_*.py`
- `parity_direction_intervene.py`, `parity_direction_sweep.py`
- `mechanistic_depth_analysis.py`, `mem_dim_breakdown.py`
- `subspace_orthogonality_over_time.py`
- `plot_paper_figures.py`
- `run_phase_256.py`, `run_full_256.py`, `run_dynamic_range.py`
- `run_adaptive_floating.py`, `run_mild_envelope.py`, `run_static_*.py`, `run_weight_tracking.py`
- `diagnose_14025.py`

## Working-store binaries (not in git; connector is text-first)

Copy these with a local `git add` using the recipe in `papers/README.md`.

### Papers / figures
- `Residual_Stream_Component_Grokking_v6_full.pdf`
- `fig5_capacity_scaling.png` … `fig9_concentrated_vs_distributed.png`

### Result archives
- `residual_sweep_results.tar.gz`
- `dual_interference_4_24.tar.gz`
- `dual_80seeds_results.tar.gz`
- `wider_dyck_40seeds_results.tar`
- `deep_wide_dyck_40seeds_results.tar`
- `gpt2_dit_results.tar` / `gpt2_dit_logs.tar`
- `gpt2_multi_results.tar` / `gpt2_multi_logs.tar`
- `gpt2_joint4_results.tar` / `gpt2_joint4_logs.tar`
- `gpt2_sae_results.tar` / `gpt2_sae_logs.tar`
- Dyck 30-seed checkpoints `seed_100.pt`–`seed_129.pt` (~1.6MB each)

## Headlines

- Pure Dyck dual-interference: 80/80 collapse, mean min_acc = 0.0016, mean min_correct_logit = −1.93
- Capacity: 128 80/80, 256 40/40, 512×4 40/40 (extracted metas; worst 512×4 seed min_acc = 0.010)
- GPT-2 Dyck-in-Text dual: 0/24 collapse, mean min_acc = 0.40, mean final_acc = 0.60
- GPT-2 joint-4: 0/8 collapse, mean best_acc = 0.40
- Contrastive cancel, Dyck 128×2, 30 seeds: 9/30 strong collapse (≤0.03); controls intact
