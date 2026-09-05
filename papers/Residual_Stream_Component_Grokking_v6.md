# A Causally Necessary Residual-Stream Component in Grokking, and Its Geometry in Language-Model Residual Streams

**Tony Taddonio**  
Independent researcher · t2addonio@gmail.com

Version 6 — July 2026

Incorporates capacity scaling, dual- and multi-direction residual interference, and transfer to a pretrained language-model residual stream.

---

## Abstract

We identify a low-dimensional residual-stream component in small transformers that is recoverable early via a train-versus-held-out contrast, remains causally necessary after generalization, and is nearly orthogonal to previously described algorithmic features. Phase-cancelling this component collapses test accuracy; matched random and algorithmic controls do not. The pattern replicates across modular addition and multiplication, sparse parity, sparse majority, and the Dyck language. On sparse parity a two-dimensional high-energy core is necessary and sufficient; on Dyck language the causal effect concentrates, in a substantial fraction of seeds, in a single direction maximally activated by locally illegal prefixes.

We extend the intervention from single-direction phase cancellation to full phase–magnitude sweeps, dual-direction residual interference, sequential multi-direction cancellation, and simultaneous joint multi-direction cancellation. On pure algorithmic models these procedures recover near-complete causal collapse even after substantial capacity scaling (width 512, depth 4). When the same hierarchical decision is placed inside a pretrained language-model residual stream (GPT-2 fine-tuned on Dyck-in-Text), residual support becomes distributed: individual directions and small combinations remain causally relevant, yet linear residual interventions no longer produce near-total collapse. We interpret the contrast as a change in residual geometry—from concentrated and low-rank in from-scratch algorithmic models to distributed and only partially linear in pretrained language-model residual streams—and position residual phase/magnitude analysis as a practical tool for causal discovery of residual primitives under both regimes.

## 1. Introduction

Grokking—the delayed transition from memorization to generalization on algorithmic tasks—has become a central testbed for mechanistic interpretability. The dominant account (Nanda et al., 2023) reverse-engineers a Fourier-based circuit for modular addition.

Prior analyses focus primarily on the generalizing circuit itself. Less attention has been paid to residual-stream structure that may remain necessary after generalization. Using contrastive extraction (train versus held-out activations), we recover a low-dimensional subspace that can be identified early, stays causally necessary after high test accuracy, and is nearly orthogonal to known algorithmic features.

We then treat residual directions as signals and introduce phase rotation, magnitude modulation, dual-direction interference, and multi-direction sequential and joint cancellation. On algorithmic models these procedures convert residual ablation into a high-reliability causal discovery method. When transferred to a pretrained language-model residual stream the same formal decision is supported by a more distributed collection of weaker directions.

## 2. Related Work

Nanda et al. (2023) reverse-engineered the Fourier multiplication circuit for modular addition. Our results are compatible with an early memorization phase later removed by weight decay, but show that a distinct residual-stream structure persists and remains load-bearing after generalization.

More recent residual-stream / SAE work has emphasized distributed and polysemantic features in pretrained language models. The transfer experiments sit at that intersection: a residual structure isolated in the clean algorithmic setting is asked how its causal geometry changes inside a pretrained LM residual stream.

## 3. Methods

**Models and tasks.** 2–4 layer decoder-only transformers with d_model ∈ {128, 256, 512}. Algorithmic tasks: modular addition and multiplication (p ∈ {113, 257}), sparse parity (n=20, k=4), sparse majority (n=20, k=5), Dyck language. LM transfer: GPT-2 Small, Dyck-in-Text.

**Contrastive extraction.** Difference of class-conditional residual means at the final layer seeds a low-rank SVD. Top-k right singular vectors are QR-orthonormalized.

**Phase and magnitude.** A unit direction u is rotated in the plane spanned by u and an orthogonal complement o: û(θ) = cos θ · u + sin θ · o. Scale s is swept independently (typically [−1.5, 3.5]).

**Dual-direction interference.** Orthonormal pair (u1, u2), relative phase Δθ, independent scales s1, s2.

**Multi-direction sequential and joint cancellation.** Pool of eight candidate directions (leading singular vectors + contrastive + mean). Greedy ranking by accuracy drop; top-4 cancelled simultaneously on a scale grid (5^4 = 625).

**Hook.** `blocks.{L-1}.hook_resid_post`. Controls: matched-rank random; Fourier / stack-depth probes.

## 4. Results

### 4.1 Original residual component

On modular addition and multiplication the contrastive component is recoverable early, remains causally necessary after high test accuracy, and is nearly orthogonal to Fourier features. Phase cancellation drives held-out accuracy to chance; matched random and Fourier controls do not. Sparse parity: a 2-D high-energy core is necessary and sufficient. Dyck: strong collapse in about one-third of seeds under 1-D cancellation, concentrated on a direction maximally activated by locally illegal prefixes.

### 4.2 Capacity scaling

Holding Dyck fixed: width 128 → 256 (2 layers) → 512×4. Dual-direction interference recovered near-complete collapse in all three regimes (see `results/SUMMARY.md`): 80/80 at width 128, 40/40 at width 256, 40/40 at 512×4 in the extracted metas (mean min_acc ≤ 0.002).

### 4.3 Dual-direction residual interference

On the original 128-d models a systematic dual-direction search produced full causal collapse on every one of 80 seeds. At the optimal point the correct-class logit was driven negative (mean ≈ −1.93).

### 4.4 Transfer to a pretrained LM residual stream

GPT-2 Small fine-tuned on Dyck-in-Text saturated at 0.55–0.63 held-out accuracy. Residual directions still moved behavior: strongest single direction typically dropped accuracy by 0.15–0.28. Joint-4 bottomed in the 0.37–0.42 range. No near-zero collapse.

### 4.5 Concentrated versus distributed geometry

From-scratch algorithmic transformers: low-rank, nearly fully ablatable residual support. Pretrained LM residual stream: distributed weaker directions; joint linear cancellation remains incomplete.

## 5. Discussion

The toolkit converts residual ablation into systematic causal discovery. Collapse-to-zero is a strong diagnostic when the representation is low-rank. When it is not, the value is ranking residual primitives and measuring their individual and joint effects.

Limitation: GPT-2 never acquired a strong hierarchical capability, so the distributed result is lower-bounded by weak task learning.

## 6. Conclusion

A low-dimensional residual-stream component remains causally necessary after grokking and is distinct from known algorithmic features. Phase / magnitude / dual / multi-direction interference turns that component into a practical discovery tool. Transfer to a pretrained LM residual stream reveals a change in geometry—from concentrated to distributed—while preserving the causal relevance of the extracted directions.

## 7. Reproducibility

PyTorch + TransformerLens. Interventions are evaluation-time forward hooks. Multi-seed runs used one seed per GPU on 8×4090-class hardware. Scripts live in this repository (`train_dyck*.py`, `analyze_dyck_directions.py`, `experiments/joint4/`).

## Appendix A — Hyperparameters

See also `results/SUMMARY.md`.

**Algorithmic models:** d_model 128/256/512; n_layers 2–4; n_heads 4–16; AdamW; lr 1e-3–3e-4; weight decay 0.1–1.0; batch 256–512; seeds 30–80.

**Interventions:** phase 0–360° step 3–15°; scale [−1.5, 3.5]; dual Δθ step 10–15°; joint-4 scale grid 5 values.

**GPT-2 transfer:** gpt2 124M; context 128; fine-tune 3k–4k steps; batch 16; lr 2e-5; 8 seeds for multi-direction and joint-4.
