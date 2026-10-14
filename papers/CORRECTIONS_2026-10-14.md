# Errata and corrections draft — t2addonio research repos (2026-10-14)

Draft for author review. Nothing in the PDFs has been edited. Every number below comes from one of:

- **[VR]** `/workspace/verification-report.md` (audit of all 5 repos, 2026-10-14)
- **[RR]** `/workspace/research/runs/rerun-summary.md` and the run folders under `/workspace/research/runs/2026-10-14-*` (CPU re-runs on merged `residual-stream-grokking@6848a00`); copied into `residual-stream-grokking/results/rerun-2026-10-14/`
- **[CSV]** `residual-stream-grokking/results/csv/*.csv` and `results/dyck_30seed/*.json`
- **[PR]** fix PRs merged 2026-10-14: RSG #1/#2, RCT #1/#2, IRG #1, hybrid-lab #6/#7

Status key: **corrected** (replace with the stated text) · **weakened** (claim stays, but narrower/caveated) · **needs rerun** (current evidence insufficient; rerun with fixed code before keeping) · **unbacked — remove or supply data** (no script/data in the repo supports it).

Paper files: v5 = `residual-stream-grokking/Residual_Stream_Component_Grokking.pdf` (8 pp.); v6 = `residual-stream-grokking/papers/Residual_Stream_Component_Grokking_v6_full.pdf` (10 pp.) and its markdown `papers/Residual_Stream_Component_Grokking_v6.md`. Quotes were taken with `pdftotext -layout`.

---

## A. v5 paper (`Residual_Stream_Component_Grokking.pdf`)

### A1. Random control "essentially intact" — §4.1, p. 2 (also Abstract p. 1, Fig. 1–2 captions, Table 1 p. 8)
- **Original (§4.1):** "On modular addition and multiplication, cancelling the contrastive subspace at coefficient 2.0 drives test accuracy to 0.00, while Fourier/multiplicative and random controls leave performance essentially intact (Figures 1–2)."
- **Original (Abstract):** "Phase-cancelling this component collapses test accuracy to zero; matched random and algorithmic controls do not."
- **Original (Fig. 1 caption):** "Contrastive cancel → accuracy 0.00; Fourier and random controls do not." **Fig. 2:** "Identical selective collapse." **Table 1:** `random c=2` column "~baseline" for every row.
- **Evidence [RR §3]:** 5 seeds (0–4) per width/task, c=2, original settings. Random-subspace cancel: addition d128 **0.403 ± 0.175**, multiplication d128 **0.423 ± 0.204** (every seed 0.18–0.76); addition d256 0.985 ± 0.017, multiplication d256 0.947 ± 0.086 (seed 4: 0.80). Contrastive → 0.000 in all four cells; Fourier/algorithmic 0.975–1.000; overlap 0.0028–0.0084. Random c=1 at d128: add 0.858 ± 0.189, mul 0.925 ± 0.089. Single-seed audit runs agreed (d128: add 0.163, mul 0.547) [VR §4]. The random subspace has rank k_mem + k_fourier, a large share of d=128.
- **Status:** **corrected** (d128) / **weakened** (overall selectivity claim).
- **Replacement (§4.1):** see paragraph P-v5-4.1 below.
- **Replacement (Abstract sentence):** "Phase-cancelling this component collapses test accuracy to zero, while algorithmic (Fourier/multiplicative) controls leave it intact; a matched-rank random control is largely harmless at width 256 but substantially damages width-128 models (mean test accuracy ≈ 0.40–0.42 at coefficient 2), so selectivity relative to random subspaces holds only at the larger width."
- **Replacement (Fig. 1 caption):** "Causal ablation on modular addition. Contrastive cancel → accuracy 0.00; Fourier control intact. Random control intact at d=256 (0.985 ± 0.017, 5 seeds) but not at d=128 (0.403 ± 0.175)."
- **Replacement (Fig. 2 caption):** "Causal ablation on modular multiplication. Contrastive cancel → 0.00; multiplicative control intact; random control 0.947 ± 0.086 at d=256, 0.423 ± 0.204 at d=128 (5 seeds)."
- **Replacement (Table 1, random c=2 column):** Mod. addition: "0.40 ± 0.18 (d128), 0.99 ± 0.02 (d256)"; Mod. multiplication: "0.42 ± 0.20 (d128), 0.95 ± 0.09 (d256)". Other rows (p=257, sparse parity, sparse majority): "not re-run" — keep "~baseline" only with a footnote that these were not re-verified (sparse parity was too slow on CPU [VR §4]).

### A2. Weight-norm clipping — §2 Related Work, p. 1; §5 Discussion, p. 8
- **Original (§2):** "Weight-norm clipping methods accelerate grokking by constraining decoder-row magnitudes; because those weights read from the residual stream, they necessarily interact with residual-stream geometry. Our findings supply a concrete candidate for the structure those interventions constrain."
- **Original (§5):** "Independent weight-norm clipping results are consistent with having constrained related residual-stream geometry."
- **Evidence:** The in-repo clipping scripts (`run_static_fixed.py`, `run_static_1.8.py`, `run_mild_envelope.py`, `run_weight_tracking.py`) had no held-out split and defined "grokked" as final training-batch loss < 0.8 [VR §4]; fixed in RSG PR #2. Re-run with a real test split, 27 runs (9 conditions × 3 seeds) [RR §2]: unclipped controls end ≥ 0.95 test acc in **6/9** (final mean **0.90**); clipped/envelope runs in **7/18** (final mean **0.68**); Weight Envelope 0/3. No grokking delay: every run memorizes by step 500 and first reaches ≥ 0.95 by step 2000–3500 (one run 6500). All conditions, controls included, show repeated drops from ≈1.0 to ≈0.01 test accuracy and recover.
- **Status:** **corrected** for the author's own clipping results (not supported); the citation to external clipping literature may stay but must not imply in-house support.
- **Replacement (§2):** "Weight-norm clipping methods have been reported to accelerate grokking by constraining decoder-row magnitudes; because those weights read from the residual stream, they may interact with residual-stream geometry. Our own clipping experiments do not support an acceleration effect in this setting (see Erratum E3), so we do not claim a link."
- **Replacement (§5):** delete the sentence, or replace with: "In our own setting, weight-norm clipping did not accelerate or stabilise generalisation (6/9 unclipped vs 7/18 clipped runs ended at ≥ 0.95 test accuracy), so we draw no connection between clipping and the component reported here."
- **Note (unsure):** whether §2 cites only external work or the author's own scripts is not clear from the text. If only external work is meant, the minimal fix is to soften "accelerate" → "have been reported to accelerate" and drop "Our findings supply a concrete candidate…".

### A3. "Grokking" framing on modular arithmetic — throughout (title, Abstract, §4)
- **Evidence:** at 50/50 split, test accuracy already 1.000 at the first eval (step 2000) on mod-113 addition, d128/d256 [VR §4]; clipping setup (train_frac 0.3) reaches ≥ 0.95 by 2000–3500 with no delayed phase [RR §2].
- **Status:** **weakened**.
- **Replacement (add to Limitations):** "In our re-runs, generalisation on modular addition occurs without a long delayed phase at the evaluation granularity used (first test-accuracy ≥ 0.95 by step 2000–3500); we use 'grokked' to mean 'trained to high held-out accuracy', not to imply a classic delayed-generalisation curve."

### A4. Sections not re-verified (p=257, sparse parity/majority §4.2–4.3, dynamics §4.4)
- **Evidence:** not re-run (sparse parity too slow on CPU [VR §4]); `plot_paper_figures.py` redraws v5 figures from hard-coded numbers [VR §4].
- **Status:** **needs rerun** (not shown wrong; just unverified). Add a footnote: "Figures in this version were drawn from tabulated numbers; regenerating scripts for Figs. 5–11 are being added."

### Ready-to-paste paragraph P-v5-4.1 (replaces §4.1 body)
> On modular addition and multiplication, cancelling the contrastive subspace at coefficient 2.0 drives test accuracy to 0.00 at both width 128 and width 256 (5 seeds each), while cancelling the Fourier/multiplicative subspace leaves performance intact (0.975–1.000). A matched-rank random subspace is a weaker control than originally stated: at width 256 it leaves accuracy largely intact (addition 0.985 ± 0.017; multiplication 0.947 ± 0.086, one seed at 0.80), but at width 128 it substantially damages the model (addition 0.403 ± 0.175; multiplication 0.423 ± 0.204 at c=2; 0.858 ± 0.189 and 0.925 ± 0.089 at c=1). Because the random subspace has rank k_mem + k_Fourier, it occupies a large fraction of a 128-dimensional residual stream; the contrastive effect is therefore clearly selective relative to random subspaces only at width 256. Overlap between the contrastive and algorithmic subspaces is small (0.003–0.008). Causal collapse is strongest in the final residual stream (Figure 4).

---

## B. v6 paper (`papers/Residual_Stream_Component_Grokking_v6_full.pdf`, and `papers/Residual_Stream_Component_Grokking_v6.md`)

### B1. 39/40 at 512×4 — §4.2, p. 6
- **Original:** "…100 % of seeds (80/80) at width 128, 100 % (40/40) at width 256, and 97.5 % (39/40) at 512×4, with mean minimum accuracy remaining below 0.002."
- **Evidence:** `results/csv/deep_wide_dyck_40seeds.csv`: 40/40 below 0.05, max min_acc 0.0100 (seed 28), mean 0.0019 [CSV, VR §4]. (The markdown v6 already says 40/40; only the PDF is wrong.)
- **Status:** **corrected**.
- **Replacement:** "…100 % of seeds (80/80) at width 128, 100 % (40/40) at width 256, and 100 % (40/40) at 512×4 (worst seed min_acc = 0.010), with mean minimum accuracy remaining below 0.002."

### B2. Dyck 30-seed "strong collapse in approximately one-third of runs" — §4.1, p. 6; Abstract p. 2 ("in a substantial fraction of seeds, in a single direction…")
- **Original (§4.1):** "On the Dyck language, across 30 random seeds the contrastive component produces strong causal collapse in approximately one-third of runs. In those runs the effect concentrates in a single residual direction that is maximally activated by locally illegal parentheses prefixes and remains nearly orthogonal to linear probes for stack depth and open/close counts."
- **Evidence [RR §1]:** the old split (9 strong / 3 moderate / 18 ≈ chance) came from an accuracy bug (`preds % 2 == y` let the pad token count as class 0; fixed in RSG PR #2) plus an unshuffled held-out slice used for early stopping. Re-run of seeds 100–129, original settings, fixed code: contrastive cancel → **30/30 label flips**, accuracy **0.007 ± 0.014** (max 0.078), pred_pos_rate 0.499; top-1 direction alone 0.033 ± 0.099. Base 0.993 ± 0.013. Algorithmic control 0.993 ± 0.013, random control 0.992 ± 0.019. **But** the contrastive subspace aligns 0.977 ± 0.014 with the readout logit-difference direction (algorithmic 0.18, random 0.24), so with c=2 a flip is close to built-in. The algorithmic subspace is **rank 3, not 8**; 23/30 seeds give algorithmic-cancel accuracy exactly equal to base (hook near no-op). `overlap_contrastive_algo` = 2.73 ± 0.49 (> 1; normalisation bug), so old ≈0.013 overlap values are not interpretable.
- **Status:** **corrected** (count) and **weakened** (interpretation); overlap values **needs rerun** after metric fix.
- **Replacement:** "On the Dyck language (30 seeds, 2 layers, d=128), cancelling the contrastive subspace at c=2 inverts the model's decisions on every seed (held-out accuracy 0.007 ± 0.014 against a base of 0.993; a single top direction already gives 0.033 ± 0.099), while the stack/count and random controls leave accuracy intact. However, the contrastive subspace is almost aligned with the model's readout direction (alignment 0.977 ± 0.014, vs 0.18 for the algorithmic and 0.24 for the random subspace), so this flip is largely expected by construction and does not by itself show a separate component; a readout-matched control is needed. An earlier analysis reporting strong collapse on about one-third of seeds was an artefact of an accuracy-metric bug and is withdrawn. The algorithmic control subspace has rank 3 (not 8), and the reported overlap values are withdrawn pending a fix to the overlap normalisation."
- **Note (unsure):** the claim that the top direction is "maximally activated by locally illegal prefixes" and "nearly orthogonal to stack-depth probes" was not re-checked in the re-run. Kept out of the replacement; author should re-add only if re-verified.

### B3. Dual-interference / capacity collapse interpretation — §4.2–4.3, pp. 6–7
- **Original (§4.3):** "…This establishes that, for these algorithmic models, a two-dimensional residual subspace is sufficient to capture the load-bearing decision variable."
- **Evidence:** numbers match the CSVs (80/80, mean min_acc 0.0016, mean min_correct_logit −1.93) [VR §4]. In a balanced binary task, min_acc ≈ 0 means the labels are inverted; by analogy to B2 (readout alignment 0.977), an intervention aligned with the readout will flip labels almost by construction. Training scripts for dual-80 / 256 / 512×4 are not in the repo [VR §4].
- **Status:** numbers **corrected/OK**; interpretation **weakened**; reproducibility **unbacked — supply scripts**.
- **Replacement (§4.3 last sentence):** "In a balanced binary task, accuracy near 0 means the model's decision was inverted rather than erased. This shows that a two-dimensional residual subspace can control the decision variable, but we have not yet tested whether that subspace is more than the readout direction itself (see §4.1). The training and search scripts for these runs are not yet in the public repository."

### B4. GPT-2 Dyck-in-Text — §4.4, p. 8 (and Abstract p. 2, §4.5 p. 9, §5 p. 10)
- **Original:** "GPT-2 Small was lightly fine-tuned on Dyck-in-Text. Fine-tuning accuracy saturated between 0.55 and 0.63, indicating that the hierarchical decision was only weakly acquired. Despite the modest task performance, residual directions extracted from the final stream retained measurable causal effect (Figures 7–8). The strongest single direction typically reduced accuracy by 0.15–0.28. The best pairwise combination and a simultaneous four-direction cancellation produced further but limited improvement, bottoming in the 0.37–0.42 range."
- **Evidence [CSV, VR §4]:** 24 seeds, final_acc 0.5507–0.6227 (mean 0.5963), min_acc 0.3667–0.4153; final_acc + min_acc = 0.94–1.01 on every seed, i.e. the "drop" is the intervention flipping a near-chance binary classifier. Joint-4 best_acc 0.3683–0.4167 (8 seeds). No single-direction or sequential data or script exists in the repo for "0.15–0.28" or Fig. 7.
- **Status:** range **corrected** (0.55–0.62 in data, paper says 0.63); causal-effect claim **weakened**; single-direction 0.15–0.28 and sequential claims **unbacked — remove or supply data**.
- **Replacement:** see paragraph P-v6-4.4 below. Also, the abstract's "residual support becomes distributed: individual directions and small combinations remain causally relevant, yet linear residual interventions no longer produce near-total collapse" → "the fine-tuned model learns the task only weakly (held-out accuracy 0.55–0.62), and residual interventions mostly flip its near-chance predictions; we treat this transfer as preliminary."

### B5. §4.5 "Concentrated versus distributed" and §5 Discussion — pp. 9–10
- **Original (§4.5):** "In the pretrained language-model residual stream the same formal decision is supported by a more distributed collection of weaker directions whose joint linear cancellation remains incomplete (Figure 9)."
- **Evidence:** follows from B3 + B4: the from-scratch side is likely readout-aligned flipping, the GPT-2 side is a near-chance classifier. Fig. 9 has no generating script.
- **Status:** **weakened** (concentrated-vs-distributed is a hypothesis, not a finding).
- **Replacement:** "We hypothesise that residual support for the decision is low-rank in from-scratch models and more distributed in a pretrained language model. The current evidence does not separate this from two simpler explanations: the from-scratch interventions are nearly aligned with the readout, and the GPT-2 model learned the task only weakly. Testing the hypothesis needs a readout-matched control and a GPT-2 model that learns the task well."

### B6. Figures 5–9 and §4.4 data — pp. 6–9; `papers/FIGURES.md`
- **Evidence:** no generating scripts; dual-interference, capacity-scaling and GPT-2 multi-direction training scripts missing; `reproduce.sh` marks them TODO; `experiments/joint4/run_joint4.py` needs `dirs.npy` not in the repo; figures and tars live in the "working store" [VR §4].
- **Status:** **unbacked — remove or supply data/scripts**. Figs. 5–6 are consistent with committed CSVs; Figs. 7–8 (single/sequential) have no committed data at all.
- **Replacement (add to §7 Reproducibility):** "Per-seed results for Figures 5, 6 and the GPT-2 dual/joint-4 runs are in `results/csv/`. The scripts that trained these models and produced Figures 5–9, and the single-direction and sequential GPT-2 data behind Figure 7 and the 0.15–0.28 range in §4.4, are not yet public; until they are, those results should be treated as unreproduced."

### B7. Weight-norm clipping — §2 Related Work, p. 4
- Same text and correction as A2. **Status:** **corrected**.

### B8. §4.1 modular-arithmetic random control — p. 6 ("matched random and Fourier controls do not") and Abstract p. 2
- Same evidence as A1. **Status:** **corrected/weakened**. Replacement: "Phase cancellation drives held-out accuracy to chance; the Fourier control does not. A matched-rank random control is harmless at width 256 but substantially damaging at width 128 (≈0.40 mean accuracy at c=2), see v5 erratum."

### Ready-to-paste paragraph P-v6-4.1 (replaces §4.1)
> On modular addition and multiplication the contrastive residual component is recoverable early in training, remains causally necessary after the model reaches high test accuracy, and is nearly orthogonal to the Fourier features of the known generalising circuit. Phase cancellation of the component drives held-out accuracy to 0.00, and the Fourier control leaves it intact. A matched-rank random control leaves width-256 models largely intact (0.985 ± 0.017 addition, 0.947 ± 0.086 multiplication; 5 seeds) but substantially damages width-128 models (0.403 ± 0.175 and 0.423 ± 0.204), so selectivity against random subspaces holds only at the larger width. On the Dyck language (30 seeds, d=128), contrastive cancellation at c=2 inverts the decision on every seed (0.007 ± 0.014 vs base 0.993), while stack/count and random controls leave accuracy intact. Because the contrastive subspace is almost aligned with the readout direction (0.977 ± 0.014), this inversion is largely expected by construction and is not, on its own, evidence for a separate component. An earlier report of strong collapse on about one-third of Dyck seeds came from an accuracy-metric bug and is withdrawn.

### Ready-to-paste paragraph P-v6-4.2 (replaces §4.2 body)
> Holding the Dyck task fixed, width was scaled from 128 to 256 (2 layers) and then to a joint width–depth increase of 512×4. Dual-direction interference drove held-out accuracy below 0.05 in all three regimes (Figure 5): 80/80 seeds at width 128, 40/40 at width 256 and 40/40 at 512×4 (worst seed 0.010), with mean minimum accuracy below 0.002. Because the task is balanced and binary, these values mean the decisions were inverted; see §4.1 for why this may largely reflect alignment with the readout direction.

### Ready-to-paste paragraph P-v6-4.4 (replaces §4.4 body)
> GPT-2 Small was lightly fine-tuned on Dyck-in-Text. Held-out accuracy reached only 0.55–0.62 across 24 seeds (mean 0.60) on this two-class task, so the hierarchical decision was barely acquired. Dual-direction residual interference lowered accuracy to 0.37–0.42 (mean 0.40), and simultaneous four-direction cancellation to 0.37–0.42 (mean 0.40, 8 seeds). On every seed, final accuracy plus post-intervention accuracy is 0.94–1.01, which is what flipping the predictions of a near-chance classifier would produce. We therefore treat these results as weak evidence of a causal effect. Earlier drafts reported single-direction drops of 0.15–0.28 and a sequential-cancellation profile (Figure 7); the data and scripts behind those are not public and the claims are withdrawn until they are.

---

## C. residual-stream-grokking repo docs

| # | Location | Original | Correction | Evidence | Status |
|---|---|---|---|---|---|
| C1 | `results/dyck_30seed/README.md` Headlines | "Strong collapse (cancel acc ≤ 0.03): **9/30** … Chance / near-chance (≥ 0.4): 18 seeds" | Mark superseded; new headline 30/30 label flip, 0.007 ± 0.014, readout alignment 0.977, rank-3 algo, overlap column invalid | RR §1 | corrected (old table kept, marked superseded) |
| C2 | `results/SUMMARY.md` / `results/README.md` Headlines row "Contrastive cancel, Dyck 128×2: 9/30 strong; 18/30 ~chance" and interpretation "seed-variable on Dyck 128×2 (9/30)" | same | 30/30 label flips (rerun); seed-variability claim withdrawn | RR §1 | corrected |
| C3 | `results/README.md` interpretation "concentrated / low-rank … distributed support on weakly fine-tuned GPT-2" | same | hypothesis only; GPT-2 near chance; readout caveat | B4–B5 | weakened |
| C4 | `INDEX.md` Headlines "9/30 strong collapse (≤0.03); controls intact" | same | 30/30 label flip (rerun 2026-10-14) | RR §1 | corrected |
| C5 | `README.md` one-line claim "collapse it on from-scratch algorithmic models … distributed and only partially linear" | same | add caveats (random control at d128, readout alignment, GPT-2 near chance) | A1, B2, B4 | weakened |
| C6 | `INDEX.md` clipping scripts `run_static_*.py`, `run_mild_envelope.py`, `run_weight_tracking.py` | listed without caveat | results from before RSG PR #2 invalid (no test split); rerun: no benefit from clipping | RR §2, VR §4 | corrected |
| C7 | `papers/Residual_Stream_Component_Grokking_v6.md` §4.1, §4.4 | see B2, B4 | apply P-v6-4.1, P-v6-4.4 with erratum banner | B2, B4 | corrected |
| C8 | `papers/FIGURES.md` "Generated 2026-07-31 from … archives" | same | note: no generating scripts in repo; Figs 7–8 data not public | B6 | unbacked |

## D. isolate-rescue-grokking

| # | Location | Original | Correction | Evidence | Status |
|---|---|---|---|---|---|
| D1 | `INDEX.md` lines 23, 25 | "`kits/isolate-rescue-vast/worker.py` \| Train modular add, …" ; "`kits/joint-residual/worker.py` \| Same + …" | mark "**not in repo** (working store only)" | VR §6 | unbacked — supply worker.py |
| D2 | `INDEX.md` lines 34–39 (results JSONs: `summary16.json`, `posthoc_summary.json`, `between_combo_summary.json`, `phase16_summary.json`, `summary_from_logs.json`, `per_seed_science.json`) and layout `results/streams_run16/` etc. | listed as present | mark "not in repo" | VR §6 (`results/` holds only HEADLINES.md) | unbacked — supply data |
| D3 | `results/HEADLINES.md` line 3 | "Source files in the working store; compact copies live next to this file when uploaded." | "None of the source files are in this repository yet; every number below is unverified from the repo." | VR §6 | unbacked |
| D4 | `README.md` "Headline findings (16-seed, 2026-08-16)" and Layout listing `results/streams_run/`… | stated as findings | label as unverified: no training code, no data, seeding unconfirmed | VR §6 | unbacked — remove or supply data |
| D5 | `STATUS.md` line 5 | "Kits + compact science JSONs uploaded" | "Kits uploaded without `worker.py`; science JSONs not uploaded" | VR §6 | corrected |
| D6 | `README.md` "All 16 seeds grokked (`base ≈ 1.0`)" vs HEADLINES `keep_all` down to 0.8785 | — | not a contradiction (different metrics); clarify that `keep_all` ≠ base | VR §6 | clarified |

## E. residual-causal-toolkit

| # | Location | Original | Correction | Evidence | Status |
|---|---|---|---|---|---|
| E1 | `README.md` Status, `INDEX.md` line 27, `docs/papers/*Progress*` l.17, `*Unified*` l.59 | "GPT-2 Phases A–D (16/16 online write-hook)" | "not reproducible from this repo: `phase_a_frozen_ports` / `phase_c_closed_loop` modules missing; no Phase D results committed" | VR §2, §5 | unbacked — remove or supply data |
| E2 | `README.md` Status "real UAH Mueller", "real Hole/NoHole 14-D", "Planck residual ports", "real naval" | "Closed" | committed scripts run on planted synthetic data; real-data runs not reproducible; label as synthetic demos | VR §5 | unbacked (real) / demo (synthetic) |
| E3 | `README.md` Status "EEG", quantum, transformer 14-D | "Closed" | only `*_summary.txt` files, no scripts | VR §5 | unbacked |
| E4 | `README.md` methodology | PASS results | planted linear residuals: near-zero residual after projection (~1e-24) is true by construction; shows code correctness, not real-world validity | VR §5 | weakened |
| E5 | `experiments/telemetry/telemetry_residual_path_patch.py` (pre-PR #2) | random control seeded with `abs(hash(name))` | non-reproducible across runs; fixed to `zlib.crc32` in PR #2; any random-control numbers produced before 2026-10-14 can't be reproduced exactly | VR §5, RCT PR #2 | needs rerun |
| E6 | `README.md` Output folder | "`interferometer.py` now imports `residual_interferometer` from `packages/residual_isa_phase_e/`" | wrong (that shim self-imported); PR #2 points it to `experiments/isa/` | VR §2, RCT PR #2 | corrected |

## F. t2addonio profile README
| # | Location | Original | Correction | Status |
|---|---|---|---|---|
| F1 | "How to read the code" item 2 | "`isolate-rescue-grokking/kits/` for 16-seed isolate/rescue/phase/freeze and the additive-tributary result (…)" | "…for the isolate/rescue/phase/freeze kits. The 16-seed results reported there can't yet be checked: the training script (`worker.py`) and result files are not in the repo." | unbacked |
| F2 | Stage 1 table row | "is there a residual-stream component that is causally necessary…" | unchanged (it is a question); no edit | — |

---

## G. Short public errata note (for README / ERRATA.md / a paper-page comment)

> **Errata (14 October 2026).** A re-check of the code and data behind *A Causally Necessary Residual-Stream Component in Grokking* (v5 and v6) found several errors. (1) The matched random control is not harmless at width 128: cancelling a random subspace of the same rank lowers modular-addition and multiplication accuracy to about 0.40 (5 seeds each); it is largely harmless at width 256. The contrastive cancellation to 0.00 does reproduce. (2) v6 §4.2 should read 40/40, not 39/40, at 512×4. (3) The Dyck 30-seed "collapse in about one-third of seeds" came from an accuracy bug. With the fix, cancellation inverts the decision on all 30 seeds, but the cancelled subspace is about 98% aligned with the model's readout direction, so this inversion is largely built in and does not by itself show a separate component. The reported overlap values for Dyck are withdrawn. (4) The GPT-2 Dyck-in-Text model reached only 0.55–0.62 accuracy on a two-class task, so the reported causal effects there are weak evidence. The single-direction and sequential numbers (§4.4, Fig. 7) have no public data and are withdrawn until they do. (5) Figures 5–9 have no public generating scripts yet. (6) The in-repo weight-norm clipping scripts had no held-out test set. Re-run properly, clipping did not help: 7/18 clipped vs 6/9 unclipped runs ended at ≥ 0.95 test accuracy. The companion repo's 16-seed isolate/rescue results, and the toolkit's GPT-2 Phases A–D and real-data results, can't currently be reproduced from the public repos. Full details: `ERRATA.md` in each repository.

---

## H. Items where the correct wording is uncertain (author decision)
1. A2/B7: whether the clipping sentence refers to external literature or to the author's own runs.
2. B2: whether to keep the "illegal-prefix" max-activation and probe-orthogonality description of the Dyck direction (not re-checked).
3. B3/B5: the readout-alignment caveat was measured for the 30-seed contrastive cancel only, not for the dual-interference 80/40/40 runs; applying it there is an inference.
4. A1: Table 1 rows for p=257, sparse parity and sparse majority were not re-run; I left "~baseline" with a footnote rather than changing them.
5. A3: whether to rename "grokking" in the title; I only added a Limitations sentence.
6. IRG: whether to keep the headline numbers with an "unverified" banner (done) or remove them until data is committed.
