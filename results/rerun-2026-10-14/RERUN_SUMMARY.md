# CPU re-runs on merged main (residual-stream-grokking @ 6848a00) — 2026-10-14
Venv /workspace/venvs/rsg (torch 2.14.1, transformer-lens 3.9.0). CPU only, 8 cores, OMP_NUM_THREADS=1 per process. Nothing committed or pushed.
Plots: plots/dyck30_cancel.png, plots/v5_random_control.png, plots/clipping_test_acc.png (partial). Aggregation: make_plots.py.

## 1. Dyck 30-seed (seeds 100–129): DONE
Original settings (2L, d128, 30k max steps, balanced early stop >0.90 after 5k steps, coeff 2, layer-1 resid_post, K=8). Nothing reduced.
Runtime: training 32 min wall (6 in parallel, about 5–6 min/seed); analyze_dyck_directions 11.6 min.
Outputs: 2026-10-14-dyck30/{results,analysis,summary.txt,old_committed_summary.txt}

| | old committed (results/dyck_30seed) | rerun, fixed code |
|---|---|---|
| contrastive cancel | 10 label flip, 3 partial, 17 "chance" (artifact of the pad-token metric bug) | **30/30 label flip** (acc 0.007 ± 0.014, max 0.078; pred_pos_rate 0.499) |
| one-class collapse / partial / no effect | 17 / 3 / 0 | **0 / 0 / 0** |
| algorithmic control | 30 intact | 30 intact (0.993 ± 0.013); rank 3, not 8 |
| random control (k=8) | 30 intact | 30 intact (0.992 ± 0.019; worst seed 118: 0.891 vs base 0.924) |
| base held-out acc | 0.994 | 0.993 ± 0.013 (seed 118: 0.924) |

Steps trained: 6034 ± 183. analyze_dyck_directions: every seed is label_flip. Cancelling only the top contrastive direction already gives 0.033 ± 0.099 (top2: 0.007).
**Caveats:**
- The contrastive subspace contains about 98% of the readout logit-difference direction (readout alignment 0.977 ± 0.014; algorithmic 0.18, random 0.24). With c=2 the "cancel" reflects the class logit, so a label flip is mostly what you'd expect by construction. It isn't evidence of a separate "memorization" component. A matched-readout control is needed before any paper claim.
- `overlap_contrastive_algo` comes out at 2.73 ± 0.49, which is above 1, so the overlap normalization is wrong (it divides by min(k)=3 while summing over 8 directions). The old 0.013 figures aren't comparable. Needs a code fix.
- 23/30 seeds give algorithmic-cancel accuracy exactly equal to base. That suggests the algorithmic hook is close to a no-op.

## 3. v5 random control, c=2: DONE (5 seeds, 0–4, per width and task)
run_discovery_check.py (addition) and run_discovery_multiplication.py with original settings. All 20 runs grokked (addition d128 seed 1 stopped at test acc 0.874 because of the script's loss<0.5 early stop).
Runtime: about 5–6 min per run, 2 in parallel, 27 min wall. Outputs: 2026-10-14-v5-randomctrl/

| test acc after cancel, c=2 (mean ± sd) | add d128 | add d256 | mul d128 | mul d256 |
|---|---|---|---|---|
| random | **0.403 ± 0.175** | 0.985 ± 0.017 | **0.423 ± 0.204** | 0.947 ± 0.086 (seed 4: 0.80) |
| contrastive (mem) | 0.000 | 0.000 | 0.000 | 0.000 |
| Fourier / algorithmic | 0.975 ± 0.056 | 0.999 | 1.000 | 1.000 |
| overlap | 0.0071 | 0.0028 | 0.0084 | 0.0040 |

Random c=1: add d128 0.858 ± 0.189, mul d128 0.925 ± 0.089.
**vs v5 paper:** "random controls essentially intact" fails at d128 for both tasks, on every seed (0.18–0.76). It mostly holds at d256, but multiplication seed 4 drops to 0.80. The random subspace has rank k_mem + k_fourier, which is a much larger share of d=128. Contrastive collapse to 0 and overlap ≤ 0.015 reproduce.

## 2. Clipping scripts: DONE (27/27 runs, finished 10:44 UTC)
Settings: original (p=113, d128, 20,001 steps, batch 512, train_frac 0.3, eval every 500, grok = test acc >= 0.95). 3 seeds per condition.
Runtime: about 23 min of CPU per seed, about 10.4 core-hours in total. Wall time was much longer because the box was suspended between turns.
Curves: plots/clipping_test_acc.png. Per-seed table: run `/workspace/venvs/rsg/bin/python make_plots.py`. Raw data: 2026-10-14-clipping/out/*/*.json

"Drops" counts eval-to-eval falls from >=0.95 to <0.8. A run counts as "collapsed" if its final test acc ends more than 0.2 below its peak.

| script | condition | final test acc mean ± sd | >=0.95 at end | grok step mean (range) | collapsed runs | drops (total) |
|---|---|---|---|---|---|---|
| run_static_fixed | Static 1.8 | 0.553 ± 0.466 | 1/3 | 2333 (2000–2500) | 2/3 | 17 |
| run_static_1.8 | Control | 0.734 ± 0.393 | 1/3 | 3000 (2500–3500) | 1/3 | 13 |
| run_static_1.8 | Static 1.8 | 0.721 ± 0.476 | 2/3 | 2167 (2000–2500) | 1/3 | 14 |
| run_mild_envelope | Control | 1.000 ± 0.001 | 3/3 | 2333 (2000–3000) | 0/3 | 23 |
| run_mild_envelope | Static 2.0 | 0.800 ± 0.345 | 2/3 | 4000 (2000–6500) | 1/3 | 8 |
| run_mild_envelope | Mild Envelope | 0.826 ± 0.156 | 1/3 | 2000 | 1/3 | 15 |
| run_weight_tracking | Control | 0.969 ± 0.053 | 2/3 | 2000 | 0/3 | 15 |
| run_weight_tracking | Static 1.8 | 0.650 ± 0.438 | 1/3 | 2167 (2000–2500) | 2/3 | 24 |
| run_weight_tracking | Weight Envelope | 0.512 ± 0.449 | 0/3 | 2167 (2000–2500) | 2/3 | 15 |

**Pooled:** unclipped controls end >=0.95 in 6/9 runs (final mean 0.90). All clipped or envelope runs end >=0.95 in 7/18 (final mean 0.68).
**Findings:**
- Every run memorizes by step 500 and first reaches test acc >=0.95 by step 2000–3500 (Static 2.0 seed 4102 at 6500). There is no long delay before test accuracy rises.
- Training is unstable in every condition, controls included. Every run except Static 2.0 seed 4102 has repeated drops from about 1.0 to about 0.01 test acc, followed by recovery. Wherever the final eval lands is partly luck.
- Clipping doesn't speed up grokking and doesn't make it more stable. Clipped runs end lower and more often in a collapse. Static 1.8 is the worst (4/9 end >=0.95), and Weight Envelope ends 0/3.
- The claim the old scripts implied (clipping or an envelope helps grokking) is **not supported** under a real held-out test.
- Collapses in the controls too suggest the base recipe (lr 1e-3, wd 1.0, batch 512) is itself unstable. That needs fixing before any clipping comparison means anything. Reporting the best or mean test acc over the last N evals would be more robust than a single final eval.

## GPU job specs
None needed. Everything ran on CPU.
