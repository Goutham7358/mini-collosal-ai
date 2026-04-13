#!/usr/bin/env python3
"""Regenerate all presentation bar charts with hatch patterns for colorblind accessibility."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'hatch.linewidth': 1.5,
})


# ============================================================
# Figure 1: PCIe vs TCP (from generate_figures_phase3.py)
# ============================================================
def generate_pcie_vs_tcp():
    strategies = ['DP(4)', 'TP(4)', 'PP(4)\n1F1B', 'DP+\nZeRO-1']
    pcie_tput  = [3389, 2518, 3200, 2832]
    tcp_tput   = [1393,  959, 3070,  934]
    single_gpu = 1399

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(strategies))
    w = 0.35

    bars_pcie = ax.bar(x - w/2, pcie_tput, w, label='PCIe (intra-node)', color='#1f77b4',
                       edgecolor='black', linewidth=0.5, hatch='///')
    bars_tcp  = ax.bar(x + w/2, tcp_tput, w, label='TCP (Phase 2)', color='#d62728',
                       edgecolor='black', linewidth=0.5, hatch='...')

    for bar, val in zip(bars_pcie, pcie_tput):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_tcp, tcp_tput):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for i in range(len(strategies)):
        if tcp_tput[i] > 0:
            speedup = pcie_tput[i] / tcp_tput[i]
            ax.annotate(f'{speedup:.1f}×', xy=(x[i], max(pcie_tput[i], tcp_tput[i]) + 200),
                        ha='center', fontsize=9, fontweight='bold', color='#333333')

    ax.axhline(y=single_gpu, color='gray', linestyle='--', linewidth=1,
               label=f'Single GPU ({single_gpu:,} tok/s)', alpha=0.7)
    ax.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
    ax.set_title('PCIe vs TCP: Interconnect Impact on Parallelism Strategies', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylim(0, 4000)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig_phase3_pcie_vs_tcp.png')
    plt.close()
    print('Saved fig_phase3_pcie_vs_tcp.png')


# ============================================================
# Figure 2: GPT-2 Three Placement Strategies
# ============================================================
def generate_gpt2_three_way():
    placements = ['Good\n(PP inter-node)', 'TP inter-node', 'Bad\n(DP inter-node)']
    throughputs = [3519, 3543, 2926]
    colors = ['#4CAF50', '#FF9800', '#E53935']
    hatches = ['///', '...', 'xxx']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(placements, throughputs, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,} tok/s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    good, bad = throughputs[0], throughputs[2]
    pct = (good - bad) / bad * 100
    ax.annotate(f'Good is +{pct:.0f}% over Bad',
                xy=(2, bad - 50), xytext=(1.3, 1200),
                fontsize=10, ha='center', color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#F9A825'),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

    ax.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=12)
    ax.set_title('GPT-2 Medium (354M) — Three Placement Strategies\nSetup: dp=2, pp=2, tp=2, 8 GPUs across 2 nodes',
                 fontsize=12)
    ax.set_ylim(0, max(throughputs) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_gpt2_three_way.png')
    plt.close()
    print('Saved fig_ppt_gpt2_three_way.png')


# ============================================================
# Figure 3: T5 Three Placement Strategies
# ============================================================
def generate_t5_three_way():
    placements = ['Good\n(PP inter-node)', 'TP inter-node', 'Bad\n(DP inter-node)']
    throughputs = [5790, 5838, 4558]
    colors = ['#4CAF50', '#FF9800', '#E53935']
    hatches = ['///', '...', 'xxx']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(placements, throughputs, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.8)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,} tok/s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    good, bad = throughputs[0], throughputs[2]
    pct = (good - bad) / bad * 100
    ax.annotate(f'Good is +{pct:.0f}% over Bad',
                xy=(2, bad - 50), xytext=(1.3, 1800),
                fontsize=10, ha='center', color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#F9A825'),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

    ax.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=12)
    ax.set_title('T5-base (237M) — Three Placement Strategies\nSetup: dp=2, pp=2, tp=2, 8 GPUs across 2 nodes',
                 fontsize=12)
    ax.set_ylim(0, max(throughputs) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_t5_three_way.png')
    plt.close()
    print('Saved fig_ppt_t5_three_way.png')


# ============================================================
# Figure 4: Multi-Model Scaling
# ============================================================
def generate_model_scaling():
    models = ['GPT-2 Small\n(117M)', 'GPT-2 Medium\n(354M)', 'T5-base\n(237M)']
    single_gpu = [3879, 1404, 3359]
    hybrid_8gpu = [8678, 3519, 5790]
    speedups = [2.24, 2.51, 1.72]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, single_gpu, width, label='Single GPU', color='#90CAF9',
                   edgecolor='black', linewidth=0.8, hatch='...')
    bars2 = ax.bar(x + width/2, hybrid_8gpu, width, label='8-GPU Hybrid (Good placement)',
                   color='#1565C0', edgecolor='black', linewidth=0.8, hatch='///')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, color='#555')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, color='#333',
                fontweight='bold')

    for i, sp in enumerate(speedups):
        y_top = hybrid_8gpu[i]
        color = '#2E7D32' if sp >= 2.0 else '#E65100'
        ax.text(x[i] + width/2, y_top + 350, f'{sp}x',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color=color)

    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Hybrid Parallelism Scaling by Model\nSetup: dp=2, pp=2, tp=2, good placement',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(hybrid_8gpu) * 1.22)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('Smaller model, but scales better\nthan T5 → architectural effect',
                xy=(0 + width/2, hybrid_8gpu[0] + 350),
                xytext=(1.0, 7500),
                fontsize=9, ha='center', color='#555', style='italic',
                arrowprops=dict(arrowstyle='->', color='#999', lw=1))

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_model_scaling.png')
    plt.close()
    print('Saved fig_ppt_model_scaling.png')


if __name__ == '__main__':
    generate_pcie_vs_tcp()
    generate_gpt2_three_way()
    generate_t5_three_way()
    generate_model_scaling()
    print('\nAll bar charts regenerated with hatch patterns.')
