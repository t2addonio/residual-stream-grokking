#!/usr/bin/env python3
"""
plot_paper_figures.py
Generates the five main figures for the paper.
Zero-accuracy bars are given a small visible height + "0.00" label.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
EPS = 0.015

def visible(val):
    return EPS if val == 0.0 else val

def plot_causal_addition_256():
    seeds = ["10000", "10001", "10002"]
    baseline = [0.991, 0.804, 1.000]
    mem_c2   = [0.000, 0.000, 0.000]
    fourier  = [0.991, 0.804, 1.000]
    random   = [0.976, 0.668, 0.999]
    x = np.arange(len(seeds))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - 1.5*width, baseline, width, label="Baseline", color="#55a868")
    bars_mem = ax.bar(x - 0.5*width, [visible(v) for v in mem_c2], width, label="mem c=2.0", color="#c44e52")
    ax.bar(x + 0.5*width, fourier, width, label="Fourier c=2.0", color="#4c72b0")
    ax.bar(x + 1.5*width, random, width, label="Random c=2.0", color="#ccb974")
    for bar, val in zip(bars_mem, mem_c2):
        if val == 0.0:
            ax.text(bar.get_x() + bar.get_width()/2, EPS + 0.02, "0.00",
                    ha="center", va="bottom", fontsize=8, color="#c44e52", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_ylabel("Test Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Modular Addition — $d_{model}=256$\nCausal Ablation")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_causal_addition_256.png")
    fig.savefig(OUT_DIR / "fig1_causal_addition_256.pdf")
    plt.close(); print("Saved Figure 1")

def plot_causal_multiplication():
    seeds = ["20000", "20001", "20002", "20003"]
    baseline = [0.9995, 0.9981, 1.000, 0.9991]
    mem_c2   = [0.000, 0.000, 0.000, 0.000]
    algo     = [0.9995, 0.9981, 1.000, 0.9991]
    random   = [0.977, 0.980, 0.999, 0.983]
    x = np.arange(len(seeds)); width = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - 1.5*width, baseline, width, label="Baseline", color="#55a868")
    bars_mem = ax.bar(x - 0.5*width, [visible(v) for v in mem_c2], width, label="mem c=2.0", color="#c44e52")
    ax.bar(x + 0.5*width, algo, width, label="Algorithmic c=2.0", color="#4c72b0")
    ax.bar(x + 1.5*width, random, width, label="Random c=2.0", color="#ccb974")
    for bar, val in zip(bars_mem, mem_c2):
        if val == 0.0:
            ax.text(bar.get_x() + bar.get_width()/2, EPS + 0.02, "0.00",
                    ha="center", va="bottom", fontsize=8, color="#c44e52", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_ylabel("Test Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Modular Multiplication — $d_{model}=256$\nCausal Ablation")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_causal_multiplication.png")
    fig.savefig(OUT_DIR / "fig2_causal_multiplication.pdf")
    plt.close(); print("Saved Figure 2")

def plot_overlap_summary():
    add128 = [0.0106, 0.0093, 0.0064, 0.0066, 0.0087, 0.0097, 0.0078, 0.0065,
              0.0095, 0.0100, 0.0072, 0.0092, 0.0154, 0.0074, 0.0061]
    add256 = [0.0035, 0.0030, 0.0021]
    mult   = [0.0043, 0.0059, 0.0048, 0.0036]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.violinplot([add128, add256, mult], positions=[1, 2, 3], showmeans=True, showmedians=False)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Addition\n$d=128$", "Addition\n$d=256$", "Multiplication\n$d=256$"])
    ax.set_ylabel("Subspace Overlap with Algorithmic Features")
    ax.set_ylim(0, 0.02)
    ax.axhline(0.01, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_title("Geometric Distinctness\n(near-zero overlap with known algorithmic features)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_overlap_summary.png")
    fig.savefig(OUT_DIR / "fig3_overlap_summary.pdf")
    plt.close(); print("Saved Figure 3")

def plot_layer_sweep():
    seeds = ["12000", "12001", "12002", "12003", "12004", "12005"]
    layer0 = [0.003, 0.235, 0.035, 0.062, 0.182, 0.039]
    layer1 = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    x = np.arange(len(seeds)); width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, layer0, width, label="Layer 0 (mem c=2.0)", color="#dd8452")
    bars_l1 = ax.bar(x + width/2, [visible(v) for v in layer1], width, label="Layer 1 (mem c=2.0)", color="#c44e52")
    for bar, val in zip(bars_l1, layer1):
        if val == 0.0:
            ax.text(bar.get_x() + bar.get_width()/2, EPS + 0.01, "0.00",
                    ha="center", va="bottom", fontsize=7, color="#c44e52", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_ylabel("Test Accuracy after Ablation"); ax.set_ylim(0, 0.30)
    ax.set_title("Layer Concentration\nCausal effect is complete only in the final residual stream")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_layer_sweep.png")
    fig.savefig(OUT_DIR / "fig4_layer_sweep.pdf")
    plt.close(); print("Saved Figure 4")

def plot_cross_task_summary():
    categories = ["Addition\n(d=128)", "Addition\n(d=256)", "Multiplication\n(d=256)"]
    mem_acc    = [0.000, 0.000, 0.000]
    algo_acc   = [0.99, 0.93, 0.999]
    random_acc = [0.45, 0.88, 0.985]
    x = np.arange(len(categories)); width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars_mem = ax.bar(x - width, [visible(v) for v in mem_acc], width, label="Contrastive (mem) c=2.0", color="#c44e52")
    ax.bar(x, algo_acc, width, label="Algorithmic control c=2.0", color="#4c72b0")
    ax.bar(x + width, random_acc, width, label="Random control c=2.0", color="#ccb974")
    for bar, val in zip(bars_mem, mem_acc):
        if val == 0.0:
            ax.text(bar.get_x() + bar.get_width()/2, EPS + 0.03, "0.00",
                    ha="center", va="bottom", fontsize=9, color="#c44e52", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Mean Test Accuracy after Ablation"); ax.set_ylim(0, 1.15)
    ax.set_title("Cross-Task Causal Selectivity")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_cross_task_summary.png")
    fig.savefig(OUT_DIR / "fig5_cross_task_summary.pdf")
    plt.close(); print("Saved Figure 5")

if __name__ == "__main__":
    plot_causal_addition_256()
    plot_causal_multiplication()
    plot_overlap_summary()
    plot_layer_sweep()
    plot_cross_task_summary()
    print("\nAll main figures saved to figures/")
