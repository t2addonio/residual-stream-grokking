> **Pipeline stage 1 of 4 — Grokking / causal necessity**  
> Grokking → **causal residual isolation** → distributed/additive representations → cross-domain residual addressing → real-world signal experiments.  
> Profile map: [github.com/t2addonio](https://github.com/t2addonio) · Next: [`isolate-rescue-grokking`](https://github.com/t2addonio/isolate-rescue-grokking)

# Residual-Stream Component in Grokking

Code and index for:

**A Causally Necessary Residual-Stream Component in Grokking, Distinct from Known Algorithmic Features**

Author: Tony Taddonio ([t2addonio@gmail.com](mailto:t2addonio@gmail.com))

This is the **science-pack home** for the original grokking residual-component paper and the later capacity-scaling / dual-interference / GPT-2 transfer experiments.

## Family of repositories

| Stage | Repo | Role |
|------:|------|------|
| 1 | **[residual-stream-grokking](https://github.com/t2addonio/residual-stream-grokking)** (this repo) | Original paper + discovery / causal scripts + v6 paper index |
| 2 | **[isolate-rescue-grokking](https://github.com/t2addonio/isolate-rescue-grokking)** | Isolate / rescue / freeze-at-transition; additive tributaries |
| 3 | **[residual-causal-toolkit](https://github.com/t2addonio/residual-causal-toolkit)** | Cross-domain residual addressing (optical, RF, NV, CMB, EEG, audio, telemetry, transformers) |
| 4 | **[hybrid-lab](https://github.com/t2addonio/hybrid-lab)** | Real-world signal experiments + human-gated research bus |

## One-line claim

After grokking, a low-dimensional contrastive residual-stream component remains load-bearing and nearly orthogonal to known algorithmic features (Fourier, stack/counting). Phase / magnitude / dual-interference interventions collapse it on from-scratch algorithmic models. On a pretrained language-model residual stream the same formal decision becomes distributed and only partially linear.

## Papers

- v5 (original GitHub PDF): [`Residual_Stream_Component_Grokking.pdf`](Residual_Stream_Component_Grokking.pdf)
- v6 cohesive draft (capacity scaling + dual + multi-direction + GPT-2 transfer): see [`papers/README.md`](papers/README.md)
- Figures 5–9 (capacity, dual-80, sequential GPT-2, single/pair/joint-4, concentrated vs distributed): see [`papers/FIGURES.md`](papers/FIGURES.md)

## Layout

```
analyze_*.py / run_discovery_*.py / train_dyck*.py   original paper scripts
experiments/joint4/                                  simultaneous top-4 cancellation (GPT-2 Dyck-in-Text)
papers/                                              paper versions + figure index
INDEX.md                                             file-by-file catalog
```

## Requirements

```bash
pip install torch transformer-lens einops numpy matplotlib seaborn transformers tqdm
```

## Quick start (original Dyck / modular scripts)

```bash
python train_dyck_30seed.py
python analyze_dyck_directions.py
```

## Quick start (joint-4 GPT-2 transfer)

```bash
cd experiments/joint4
python run_joint4.py
```

Requires prior `gpt2_multi_results/` direction files (`dirs.npy` per seed).

## What is not in git

Large result archives (`.tar` / `.tar.gz`), PNG figures, and the v6 PDF live in the working artifacts store. They are catalogued in [`INDEX.md`](INDEX.md). Re-run the scripts to regenerate.

## License

MIT. See `LICENSE`.
