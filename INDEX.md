# Index — residual-stream-grokking

Catalog of this repository and of companion working-store artifacts (2026-09-05).

## Companion repositories

- https://github.com/t2addonio/residual-stream-grokking — this repo
- https://github.com/t2addonio/isolate-rescue-grokking — isolate / rescue / freeze science pack
- https://github.com/t2addonio/residual-causal-toolkit — cross-domain residual toolkit
- https://github.com/t2addonio/hybrid-lab — local multi-agent lab

## In this repo (git)

### Paper
- `Residual_Stream_Component_Grokking.pdf` — v5 paper already on GitHub
- `papers/README.md` — version map (v5 / v6)
- `papers/FIGURES.md` — Figures 5–9 captions and local paths

### Joint-4 transfer experiment
- `experiments/joint4/run_joint4.py` — 8-GPU launcher
- `experiments/joint4/worker_joint4.py` — simultaneous top-4 scale grid
- `experiments/joint4/README.md`

### Original discovery / causal scripts (root)
- `train_dyck.py`, `train_dyck_30seed.py`, `train_dyck_multiseed.py`, `dyck_data.py`
- `analyze_dyck_directions.py`
- `run_discovery_*.py` — modular add/mult, p=257, sparse parity/majority, width 256/512
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

## Working-store artifacts (not force-pushed; GitHub connector is text-first)

Local paths under `/home/workdir/artifacts` and `/home/workdir/attachments`.

### Papers / figures
| File | Notes |
|------|--------|
| `artifacts/Residual_Stream_Component_Grokking_v2.pdf` | condensed v2 synthesis |
| `artifacts/Residual_Stream_Component_Grokking_v6_full.pdf` | full cohesive v6 draft |
| `artifacts/fig5_capacity_scaling.png` | Fig 5 — collapse rate vs capacity |
| `artifacts/fig6_dual80.png` | Fig 6 — dual-interference 80-seed |
| `artifacts/fig7_sequential_gpt2.png` | Fig 7 — sequential cancellation on GPT-2 |
| `artifacts/fig8_single_pair_joint.png` | Fig 8 — single / pair / joint-4 |
| `artifacts/fig9_concentrated_vs_distributed.png` | Fig 9 — residual geometry contrast |
| `attachments/Residual_Stream_Component_Grokking.pdf` | original uploaded v5 snapshot |

### Result archives
| File | Experiment |
|------|------------|
| `attachments/residual_sweep_results.tar.gz` | 1-D phase + magnitude sweep |
| `attachments/dual_interference_4_24.tar.gz` | dual interference, resistant seeds 4 and 24 |
| `attachments/dual_80seeds_results.tar.gz` | dual interference, 80 seeds |
| `attachments/wider_dyck_40seeds_results.tar` | Dyck d_model=256, 40 seeds |
| `attachments/deep_wide_dyck_40seeds_results.tar` | Dyck 512×4, 40 seeds |
| `attachments/gpt2_dit_results.tar` / `gpt2_dit_logs.tar` | GPT-2 Dyck-in-Text dual sweep |
| `attachments/gpt2_multi_results.tar` / `gpt2_multi_logs.tar` | GPT-2 8-direction sequential + pairs |
| `attachments/gpt2_joint4_results.tar` / `gpt2_joint4_logs.tar` | GPT-2 simultaneous top-4 |
| `attachments/gpt2_sae_results.tar` / `gpt2_sae_logs.tar` | first SAE attempt (fine-tune failed) |

### Headline numbers (for the index, not a substitute for the paper)

- Pure Dyck dual-interference collapse: ~100% of 80 seeds, mean min_acc < 0.002
- Capacity: 128 (2L) 100% / 256 (2L) 100% / 512×4 ~97.5%
- GPT-2 Dyck-in-Text fine-tune accuracy ~0.55–0.63 (weak task learning)
- GPT-2 best single-dir drop ~0.15–0.28; joint-4 bottoms ~0.37–0.42
- Interpretation: concentrated residual support on from-scratch models; distributed on pretrained LM residual stream
