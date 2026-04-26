#!/usr/bin/env python3
"""
Generate Phase 3 result figures.
Primary colors (red, blue, yellow) + hatching patterns for accessibility.
No pre-fix/post-fix comparisons — only correct Phase 3 data.

Output: figures_postfix/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUT = 'figures_postfix'
os.makedirs(OUT, exist_ok=True)

# ── Primary palette ──
RED    = '#D32F2F'
BLUE   = '#1565C0'
YELLOW = '#F9A825'
RED_L  = '#EF9A9A'
BLUE_L = '#90CAF9'
YELLOW_L = '#FFF176'
BLACK  = '#212121'
GREY   = '#757575'

# ── Hatching patterns ──
H_SLASH  = '///'
H_BACK   = '\\\\\\'
H_DOT    = '...'
H_CROSS  = 'xxx'
H_HORIZ  = '---'

plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'hatch.linewidth': 1.5,
})

# ============================================================
# DATA — Phase 3 correct results
# ============================================================

# Single-GPU baselines
single_gpu_bs4 = {'GPT-2 Small': 3619, 'GPT-2 Medium': 1404, 'T5-base': 3125}
single_gpu_max = {'GPT-2 Small': (4426, 16), 'GPT-2 Medium': (1561, 8), 'T5-base': (4747, 32)}

# Placement (GPT-2 Medium, dp=2, pp=2, tp=2, bs=8)
placement_tput = {'Good': 4851, 'Bad': 4126, 'Worst': 3776}

# Batch size sweep (GPT-2 Medium, good placement)
bs_sizes    = [8, 16, 32, 64, 128]
bs_tput     = [4851, 6012, 6618, 7245, 7387]
bs_mem      = [2.57, 2.89, 3.60, 5.14, 8.21]
single_max_med = 1561

# Architecture at bs=8 (good placement, dp=2, pp=2, tp=2)
arch_1gpu   = {'GPT-2 Small': 3879, 'GPT-2 Medium': 1404, 'T5-base': 3359}
arch_8gpu   = {'GPT-2 Small': 10297, 'GPT-2 Medium': 5014, 'T5-base': 5973}

# Architecture at max batch (bs=128, good placement)
arch_8gpu_max = {'GPT-2 Small': 15546, 'GPT-2 Medium': 7387, 'T5-base': 16555}

# 8-GPU batch sweep per model (good placement)
sweep_models = {
    'GPT-2 Small':  [10297, 14534, 15509, 15546],
    'GPT-2 Medium': [5014,  6618,  7245,  7387],
    'T5-base':      [5973,  13454, 15589, 16555],
}
sweep_bs = [8, 32, 64, 128]

# Pipeline memory per stage
pp_mem = {
    'GPT-2 Medium': (2.57, 2.56),
    'T5-base': (1.69, 2.75),
}

# ZeRO-1
zero_tput    = 4405
no_zero_tput = 4851


# ================================================================
# FIGURE 1: Placement — 3-way bar chart
# ================================================================
def fig1_placement():
    placements = ['Good\n(PP inter-node)', 'Bad\n(DP inter-node)', 'Worst\n(TP inter-node)']
    tputs = [placement_tput['Good'], placement_tput['Bad'], placement_tput['Worst']]
    colors = [BLUE, YELLOW, RED]
    hatches = [H_SLASH, H_DOT, H_BACK]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(placements, tputs, color=colors, width=0.5,
                  edgecolor=BLACK, linewidth=1.2)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for bar, val in zip(bars, tputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,} tok/s', ha='center', fontsize=12, fontweight='bold', color=BLACK)

    # Percentage drops
    good = tputs[0]
    for i in [1, 2]:
        pct = (good - tputs[i]) / good * 100
        ax.annotate(f'-{pct:.1f}% vs Good',
                    xy=(i, tputs[i] - 50),
                    xytext=(i, tputs[i] - 450),
                    fontsize=10, ha='center', fontweight='bold', color=BLACK,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=BLACK, linewidth=1))

    ax.text(1.5, 600, 'Bad & Worst deadlock at bs ≥ 16',
            fontsize=10, ha='center', color=RED, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=RED_L, edgecolor=RED, linewidth=1.2))

    ax.set_ylabel('Aggregate Throughput (tok/s)', fontsize=12)
    ax.set_title('Placement Comparison — GPT-2 Medium (354M)\n'
                 'dp=2, pp=2, tp=2, bs=8, 8 GPUs across 2 nodes', fontsize=11)
    ax.set_ylim(0, max(tputs) * 1.22)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_placement.png')
    plt.close()
    print('Saved fig1_placement.png')


# ================================================================
# FIGURE 2: Batch Size Scaling (throughput bars + memory line)
# ================================================================
def fig2_batch_scaling():
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)

    # Throughput bars
    bars = ax1.bar(range(len(bs_sizes)), bs_tput, color=BLUE, alpha=0.85,
                   edgecolor=BLACK, linewidth=1.2, width=0.6, hatch=H_SLASH)
    for i, (bar, val) in enumerate(zip(bars, bs_tput)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold', color=BLUE)
        sp = val / single_max_med
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f'{sp:.2f}×', ha='center', fontsize=9, color=RED, fontweight='bold')

    # Memory line
    ax2.plot(range(len(bs_sizes)), bs_mem, 's-', color=RED, lw=2.5, markersize=8,
             markerfacecolor='white', markeredgecolor=RED, markeredgewidth=2)
    for i, m in enumerate(bs_mem):
        ax2.text(i + 0.18, m + 0.25, f'{m:.1f} GB', fontsize=9, color=RED, fontweight='bold')

    # 1-GPU max reference
    ax1.axhline(y=single_max_med, color=GREY, linestyle='--', lw=1.5)
    ax1.text(len(bs_sizes) - 0.6, single_max_med + 120, f'1-GPU max: {single_max_med:,} tok/s',
             fontsize=9, color=GREY, style='italic')

    # T4 memory limit
    ax2.axhline(y=14.5, color=RED, linestyle=':', lw=1, alpha=0.4)
    ax2.text(0, 14.8, 'T4 usable limit (~14.5 GB)', fontsize=8, color=RED, alpha=0.6)

    # Region labels
    ax1.text(0.5, 7700, 'comm-\nbound', fontsize=9, ha='center', color=GREY, style='italic')
    ax1.text(3.5, 7700, 'compute-\nbound', fontsize=9, ha='center', color=GREY, style='italic')
    ax1.annotate('', xy=(2.8, 7650), xytext=(1.5, 7650),
                 arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))

    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=12, color=BLUE)
    ax2.set_ylabel('Peak Memory per GPU (GB)', fontsize=12, color=RED)
    ax1.set_xticks(range(len(bs_sizes)))
    ax1.set_xticklabels([str(b) for b in bs_sizes], fontsize=11)
    ax1.set_ylim(0, max(bs_tput) * 1.22)
    ax2.set_ylim(0, 16)
    ax1.set_title('Batch Size Scaling — GPT-2 Medium, Good Placement\n'
                   'dp=2, pp=2, tp=2  |  1-GPU OOMs at bs=16; TP enables up to bs=128',
                   fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_batch_scaling.png')
    plt.close()
    print('Saved fig2_batch_scaling.png')


# ================================================================
# FIGURE 3: Architecture — 1-GPU vs 8-GPU at bs=8
# ================================================================
def fig3_arch_bs8():
    models = list(arch_1gpu.keys())
    sg = [arch_1gpu[m] for m in models]
    hb = [arch_8gpu[m] for m in models]
    speedups = [h / s for s, h in zip(sg, hb)]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - w/2, sg, w, label='Single GPU (bs=4)',
                   color=YELLOW, edgecolor=BLACK, linewidth=1.2, hatch=H_DOT)
    bars2 = ax.bar(x + w/2, hb, w, label='8-GPU Hybrid (bs=8, good placement)',
                   color=BLUE, edgecolor=BLACK, linewidth=1.2, hatch=H_SLASH)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{int(bar.get_height()):,}', ha='center', fontsize=9, color=BLACK)
    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold', color=BLACK)
        color = BLUE if speedups[i] >= 2.0 else RED
        ax.text(x[i] + w/2, hb[i] + 550,
                f'{speedups[i]:.2f}×', ha='center', fontsize=13, fontweight='bold', color=color)

    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Architecture Scaling at bs=8\n'
                 'dp=2, pp=2, tp=2, good placement', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, max(hb) * 1.25)
    ax.legend(fontsize=10, loc='upper right')

    # T5 vs GPT-2 Small note
    ax.annotate('Smaller model scales better\nthan T5 → architectural effect',
                xy=(0 + w/2, hb[0] + 400), xytext=(1.0, 8800),
                fontsize=9, ha='center', color=GREY, style='italic',
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_arch_bs8.png')
    plt.close()
    print('Saved fig3_arch_bs8.png')


# ================================================================
# FIGURE 4: Architecture — 8-GPU batch sweep (line chart)
# ================================================================
def fig4_arch_sweep():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    TEAL = '#00796B'
    colors = {'GPT-2 Small': BLUE, 'GPT-2 Medium': RED, 'T5-base': TEAL}
    markers = {'GPT-2 Small': 'o', 'GPT-2 Medium': 's', 'T5-base': '^'}
    lstyles = {'GPT-2 Small': '-', 'GPT-2 Medium': '--', 'T5-base': '-.'}

    for model in sweep_models:
        ax.plot(range(len(sweep_bs)), sweep_models[model],
                marker=markers[model], color=colors[model], lw=3.5, markersize=11,
                label=model, markerfacecolor='white', markeredgewidth=3,
                markeredgecolor=colors[model], linestyle=lstyles[model])
        ax.text(len(sweep_bs) - 1 + 0.15, sweep_models[model][-1],
                f'{sweep_models[model][-1]:,}', fontsize=9, color=colors[model],
                fontweight='bold', va='center')

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('8-GPU Throughput by Batch Size & Architecture\n'
                 'dp=2, pp=2, tp=2, good placement', fontsize=11)
    ax.set_xticks(range(len(sweep_bs)))
    ax.set_xticklabels([str(b) for b in sweep_bs], fontsize=11)
    ax.set_ylim(0, max(max(v) for v in sweep_models.values()) * 1.15)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax.annotate('T5 catches up at large batch\n(comm overhead hidden)',
                xy=(3, sweep_models['T5-base'][-1]),
                xytext=(1.5, 17500),
                fontsize=9, ha='center', color=TEAL, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.5))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_arch_batch_sweep.png')
    plt.close()
    print('Saved fig4_arch_batch_sweep.png')


# ================================================================
# FIGURE 5: Architecture max-batch speedup
# ================================================================
def fig5_arch_maxbatch():
    models = list(arch_8gpu_max.keys())
    sg_max = [single_gpu_max[m][0] for m in models]
    sg_bs  = [single_gpu_max[m][1] for m in models]
    hb_max = [arch_8gpu_max[m] for m in models]
    speedups = [h / s for s, h in zip(sg_max, hb_max)]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - w/2, sg_max, w, label='1-GPU (max batch)',
                   color=YELLOW, edgecolor=BLACK, linewidth=1.2, hatch=H_DOT)
    bars2 = ax.bar(x + w/2, hb_max, w, label='8-GPU (bs=128, good placement)',
                   color=BLUE, edgecolor=BLACK, linewidth=1.2, hatch=H_SLASH)

    for i, bar in enumerate(bars1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{sg_max[i]:,}\n(bs={sg_bs[i]})', ha='center', fontsize=8, color=BLACK)
    for i, bar in enumerate(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{hb_max[i]:,}', ha='center', fontsize=9, fontweight='bold', color=BLACK)
        ax.text(x[i] + w/2, hb_max[i] + 1000,
                f'{speedups[i]:.2f}×', ha='center', fontsize=13, fontweight='bold', color=RED)

    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('Max-Batch Scaling: 1-GPU best vs 8-GPU best\n'
                 'Each config at peak achievable batch size', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, max(hb_max) * 1.22)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_arch_maxbatch.png')
    plt.close()
    print('Saved fig5_arch_maxbatch.png')


# ================================================================
# FIGURE 6: Pipeline Memory Imbalance — GPT-2 vs T5
# ================================================================
def fig6_pipeline_imbalance():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['GPT-2 Medium', 'T5-base']
    s0 = [pp_mem['GPT-2 Medium'][0], pp_mem['T5-base'][0]]
    s1 = [pp_mem['GPT-2 Medium'][1], pp_mem['T5-base'][1]]

    x = np.arange(len(models))
    w = 0.3

    bars0 = ax.bar(x - w/2, s0, w, label='Stage 0 / Encoder',
                   color=BLUE_L, edgecolor=BLACK, linewidth=1.2, hatch=H_SLASH)
    bars1 = ax.bar(x + w/2, s1, w, label='Stage 1 / Decoder',
                   color=BLUE, edgecolor=BLACK, linewidth=1.2, hatch=H_BACK)

    for i, (b0, b1) in enumerate(zip(bars0, bars1)):
        ax.text(b0.get_x() + b0.get_width()/2, b0.get_height() + 0.04,
                f'{s0[i]:.2f} GB', ha='center', fontsize=10, color=BLACK, fontweight='bold')
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.04,
                f'{s1[i]:.2f} GB', ha='center', fontsize=10, color=BLACK, fontweight='bold')

    # Imbalance annotation for T5
    imb = (s1[1] - s0[1]) / s0[1] * 100
    ax.annotate(f'+{imb:.0f}% heavier\n(cross-attention)',
                xy=(1 + w/2, s1[1] + 0.02), xytext=(1.55, 3.3),
                fontsize=10, ha='center', color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    ax.text(0, max(s0[0], s1[0]) + 0.2, '≈ balanced', ha='center',
            fontsize=11, color=BLUE, fontweight='bold')

    ax.set_ylabel('Peak Memory per GPU (GB)', fontsize=12)
    ax.set_title('Pipeline Stage Memory — Symmetric vs Asymmetric\n'
                 'dp=2, pp=2, tp=2, good placement, bs=8', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(max(s0), max(s1)) * 1.45)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_pipeline_imbalance.png')
    plt.close()
    print('Saved fig6_pipeline_imbalance.png')


# ================================================================
# FIGURE 7: ZeRO-1 with 3D Parallelism
# ================================================================
def fig7_zero():
    labels = ['Without ZeRO', 'With ZeRO-1']
    tputs = [no_zero_tput, zero_tput]
    colors = [BLUE, YELLOW]
    hatches = [H_SLASH, H_DOT]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, tputs, color=colors, width=0.45,
                  edgecolor=BLACK, linewidth=1.2)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for bar, val in zip(bars, tputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,} tok/s', ha='center', fontsize=12, fontweight='bold', color=BLACK)

    overhead = (no_zero_tput - zero_tput) / no_zero_tput * 100
    ax.annotate(f'-{overhead:.1f}% overhead',
                xy=(1, zero_tput), xytext=(1.35, 3600),
                fontsize=11, ha='center', color=RED, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('ZeRO-1 with 3D Parallelism\n'
                 'GPT-2 Medium, good placement, bs=8', fontsize=11)
    ax.set_ylim(0, max(tputs) * 1.2)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_zero.png')
    plt.close()
    print('Saved fig7_zero.png')


# ================================================================
# FIGURE 8: Summary table
# ================================================================
def fig8_summary_table():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis('off')

    headers = ['Finding', 'Key Number', 'Implication']
    rows = [
        ['Best placement\n(PP on TCP)',     '+22% vs worst',           'PP tolerates slow links;\nTP and DP need fast PCIe'],
        ['Placement stability',             'Deadlock at bs≥16\nfor bad/worst', 'Correct placement is\nessential for training'],
        ['Batch sweet spot',                'bs=32–64',                '~90% peak throughput\nwith memory headroom'],
        ['TP memory savings',               '2.57 GB vs 4.07 GB\n(1-GPU)', 'Enables 16× larger\nbatch sizes'],
        ['GPT-2 vs T5 (bs=8)',             '2.65–3.57× vs 1.78×',    'Symmetric arch scales\nbetter under hybrid'],
        ['T5 catches up\n(max batch)',      '3.49× vs 3.51×',         'Large batches hide\ncomm overhead'],
        ['ZeRO-1 overhead',                 '-9.2%',                   'Acceptable; useful for\nlarger models hitting OOM'],
    ]

    cell_colors = []
    for i in range(len(rows)):
        shade = '#E3F2FD' if i % 2 == 0 else '#FFFFFF'
        cell_colors.append([shade] * 3)

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='left', colColours=[BLUE] * 3,
                     cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    for j in range(3):
        cell = table[0, j]
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)

    ax.set_title('Phase 3 Key Findings', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig8_summary_table.png')
    plt.close()
    print('Saved fig8_summary_table.png')


# ================================================================
if __name__ == '__main__':
    fig1_placement()
    fig2_batch_scaling()
    fig3_arch_bs8()
    fig4_arch_sweep()
    fig5_arch_maxbatch()
    fig6_pipeline_imbalance()
    fig7_zero()
    fig8_summary_table()
    print(f'\nAll 8 figures saved to {OUT}/')
