# Joint-4 simultaneous residual cancellation (GPT-2 Dyck-in-Text)

Loads the 8 residual directions already extracted by the multi-direction worker (`dirs.npy`) and jointly sweeps scales on the top-4 directions.

## Files

- `run_joint4.py` — 8-GPU launcher, seeds 0–7
- `worker_joint4.py` — coarse scale grid, $5^4 = 625$ combinations per seed

## Requirements

- Prior run: `gpt2_multi_results/gpu{N}/seed_{SSS}/dirs.npy`
- `torch`, `transformer-lens`, `transformers`, `tqdm`

## Launch

```bash
chmod +x run_joint4.py worker_joint4.py
python3 run_joint4.py
```

## Result (8 seeds, 2026-07-30)

Joint-4 best accuracy sat in **0.37–0.42**. No near-zero collapse. Consistent with distributed residual support on a weakly fine-tuned GPT-2 hierarchical task.
