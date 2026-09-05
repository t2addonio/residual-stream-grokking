# Papers

## v6 (canonical current draft)

- Text in repo: [`Residual_Stream_Component_Grokking_v6.md`](Residual_Stream_Component_Grokking_v6.md)
- PDF in working store: `/home/workdir/artifacts/Residual_Stream_Component_Grokking_v6_full.pdf`
- Figures 5–9: [`FIGURES.md`](FIGURES.md)
- Per-seed numbers: [`../results/SUMMARY.md`](../results/SUMMARY.md)

Title: *A Causally Necessary Residual-Stream Component in Grokking, and Its Geometry in Language-Model Residual Streams*

Adds phase × magnitude sweeps, dual-direction interference, sequential + joint-4 cancellation, capacity scaling (128 → 256 → 512×4), GPT-2 Dyck-in-Text transfer, and the concentrated-vs-distributed geometry claim.

## v5 (original GitHub PDF)

[`../Residual_Stream_Component_Grokking.pdf`](../Residual_Stream_Component_Grokking.pdf)

Core result only: contrastive residual component recoverable early, causally necessary after grokking, nearly orthogonal to Fourier / stack-counting features.

## v2 condensed synthesis

Working store: `/home/workdir/artifacts/Residual_Stream_Component_Grokking_v2.pdf`

## Binaries

The GitHub connector used here is text-first. To add the PDF, PNGs, and result tars from a machine with `git`:

```bash
git clone https://github.com/t2addonio/residual-stream-grokking.git
cd residual-stream-grokking
mkdir -p papers/figures results/archives
cp /path/to/Residual_Stream_Component_Grokking_v6_full.pdf papers/
cp /path/to/fig5_capacity_scaling.png papers/figures/
cp /path/to/fig6_dual80.png papers/figures/
cp /path/to/fig7_sequential_gpt2.png papers/figures/
cp /path/to/fig8_single_pair_joint.png papers/figures/
cp /path/to/fig9_concentrated_vs_distributed.png papers/figures/
cp /path/to/*.tar /path/to/*.tar.gz results/archives/
git add papers results/archives
git commit -m "Add v6 PDF, figures, and result archives"
git push
```
