"""Generate figures for Phase 3 report — Communication-Aware Placement & T5."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data (from results/phase3/PHASE3_RESULTS_SUMMARY.txt)
# ============================================================

# --- Figure 1: PCIe vs TCP speedup (Group 1) ---
strategies = ['DP(4)', 'TP(4)', 'PP(4)\n1F1B', 'DP+\nZeRO-1']
pcie_tput  = [3389, 2518, 3200, 2832]
tcp_tput   = [1393,  959, 3070,  934]
single_gpu = 1399

# --- Figure 2: Good vs Bad placement (Groups 2, 3) ---
placement_configs   = ['dp2×pp2×tp2', 'dp2×pp2×tp2\n+ZeRO-1']
good_placement_tput = [3503, 3044]
bad_placement_tput  = [2904, 2416]

# --- Figure 3: Three-way placement (Group 7, both models) ---
placements = ['Good\n(PP inter-node)', 'TP\ninter-node', 'Bad\n(DP inter-node)']
gpt2_placement = [3519, 3543, 2926]
t5_placement   = [5790, 5838, 4558]

# --- Figure 4: Multi-model GPT-2 placement (Group 5) ---
gpt2_models      = ['GPT-2 Small\n(117M)', 'GPT-2 Medium\n(354M)', 'GPT-2 Large\n(774M)']
gpt2_good        = [8546, 3494, 1746]
gpt2_bad         = [7156, 2926, 0]  # Large hangs
gpt2_bad_labels  = ['7,156', '2,926', 'hangs']

# --- Figure 5: Architecture scaling comparison (Group 8) ---
arch_models    = ['GPT-2 Small\n(117M)', 'GPT-2 Medium\n(354M)', 'T5-base\n(237M)']
arch_1gpu      = [3879, 1404, 3359]
arch_8gpu      = [8678, 3519, 5790]
arch_speedup   = [2.24, 2.51, 1.72]
arch_mem_s0    = [1.65, 4.07, 2.51]
arch_mem_s1    = [1.71, 4.08, 3.87]

# --- Figure 6: T5 single-node baselines (Group 6) ---
baseline_configs = ['Single GPU', 'DP(4)', 'TP(4)', 'PP(2)']
t5_baselines     = [3359, 6477, 5029, 4297]
gpt2m_baselines  = [1404, 3389, 2518, 0]  # no PP(2) for GPT-2
gpt2s_baselines  = [3879, 8546, 5458, 0]

# ============================================================
# Style constants (matching Phase 2)
# ============================================================
CLR_BASELINE = '#555555'
CLR_DP       = '#d62728'
CLR_TP       = '#2ca02c'
CLR_PP       = '#9467bd'
CLR_ZERO     = '#ff7f0e'
CLR_HYBRID   = '#1f77b4'
CLR_GOOD     = '#2ca02c'
CLR_BAD      = '#d62728'
CLR_TP_INTER = '#ff7f0e'
CLR_GPT2     = '#1f77b4'
CLR_T5       = '#d62728'

import os
os.makedirs('figures', exist_ok=True)

# ============================================================
# Figure 1: PCIe vs TCP Speedup (Grouped bar)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(strategies))
w = 0.35

bars_pcie = ax1.bar(x - w/2, pcie_tput, w, label='PCIe (intra-node)', color='#1f77b4',
                    edgecolor='black', linewidth=0.5)
bars_tcp  = ax1.bar(x + w/2, tcp_tput, w, label='TCP (Phase 2)', color='#d62728',
                    edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_pcie, pcie_tput):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars_tcp, tcp_tput):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Speedup annotations
for i in range(len(strategies)):
    if tcp_tput[i] > 0:
        speedup = pcie_tput[i] / tcp_tput[i]
        ax1.annotate(f'{speedup:.1f}×', xy=(x[i], max(pcie_tput[i], tcp_tput[i]) + 200),
                     ha='center', fontsize=9, fontweight='bold', color='#333333')

ax1.axhline(y=single_gpu, color='gray', linestyle='--', linewidth=1,
            label=f'Single GPU ({single_gpu:,} tok/s)', alpha=0.7)
ax1.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax1.set_title('PCIe vs TCP: Interconnect Impact on Parallelism Strategies', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=10)
ax1.set_ylim(0, 4000)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('figures/fig_phase3_pcie_vs_tcp.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_pcie_vs_tcp.png")

# ============================================================
# Figure 2: Good vs Bad Placement (GPT-2 Medium, with ZeRO)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
x2 = np.arange(len(placement_configs))
w2 = 0.35

bars_good = ax2.bar(x2 - w2/2, good_placement_tput, w2, label='Good (PP inter-node)',
                    color=CLR_GOOD, edgecolor='black', linewidth=0.5)
bars_bad  = ax2.bar(x2 + w2/2, bad_placement_tput, w2, label='Bad (DP inter-node)',
                    color=CLR_BAD, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_good, good_placement_tput):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars_bad, bad_placement_tput):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Delta annotations
deltas = ['-17.1%', '-20.6%']
for i in range(len(placement_configs)):
    ax2.annotate(deltas[i], xy=(x2[i] + w2/2, bad_placement_tput[i] - 100),
                 ha='center', fontsize=9, fontweight='bold', color='white')

ax2.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax2.set_title('Good vs Bad Placement — GPT-2 Medium (8 GPUs)', fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(placement_configs, fontsize=10)
ax2.set_ylim(0, 4200)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig('figures/fig_phase3_good_vs_bad.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_good_vs_bad.png")

# ============================================================
# Figure 3: Three-Way Placement — GPT-2 vs T5 (Grouped bar)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(8, 5))
x3 = np.arange(len(placements))
w3 = 0.35

bars_gpt2 = ax3.bar(x3 - w3/2, gpt2_placement, w3, label='GPT-2 Medium',
                     color=CLR_GPT2, edgecolor='black', linewidth=0.5)
bars_t5   = ax3.bar(x3 + w3/2, t5_placement, w3, label='T5-base',
                     color=CLR_T5, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_gpt2, gpt2_placement):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars_t5, t5_placement):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Good/Bad delta
ax3.annotate('+20.3%', xy=(0.15, 3700), fontsize=10, fontweight='bold', color=CLR_GPT2,
             ha='center')
ax3.annotate('+27.0%', xy=(0.55, 6100), fontsize=10, fontweight='bold', color=CLR_T5,
             ha='center')
ax3.annotate('← Good vs Bad →', xy=(1, 200), fontsize=8, ha='center', color='gray')

ax3.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax3.set_title('Three-Way Placement Comparison (dp=2, pp=2, tp=2, 8 GPUs)', fontsize=12)
ax3.set_xticks(x3)
ax3.set_xticklabels(placements, fontsize=10)
ax3.set_ylim(0, 7000)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig('figures/fig_phase3_three_way_placement.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_three_way_placement.png")

# ============================================================
# Figure 4: Multi-Model GPT-2 Placement
# ============================================================
fig4, ax4 = plt.subplots(figsize=(7, 4.5))
x4 = np.arange(len(gpt2_models))
w4 = 0.35

bars4_good = ax4.bar(x4 - w4/2, gpt2_good, w4, label='Good placement',
                     color=CLR_GOOD, edgecolor='black', linewidth=0.5)
bars4_bad  = ax4.bar(x4 + w4/2, gpt2_bad, w4, label='Bad placement',
                     color=CLR_BAD, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars4_good, gpt2_good):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val, lbl in zip(bars4_bad, gpt2_bad, gpt2_bad_labels):
    y = val if val > 0 else 200
    ax4.text(bar.get_x() + bar.get_width()/2, y + 100,
             lbl, ha='center', va='bottom', fontsize=9, fontweight='bold',
             color='black' if val > 0 else CLR_BAD)

# Delta annotations
ax4.annotate('+19.4%', xy=(x4[0], 8800), ha='center', fontsize=9, fontweight='bold', color='#333')
ax4.annotate('+19.4%', xy=(x4[1], 3800), ha='center', fontsize=9, fontweight='bold', color='#333')
ax4.annotate('hangs!', xy=(x4[2] + w4/2, 600), ha='center', fontsize=9,
             fontweight='bold', color=CLR_BAD, fontstyle='italic')

ax4.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax4.set_title('Good vs Bad Placement Across GPT-2 Sizes (8 GPUs)', fontsize=12)
ax4.set_xticks(x4)
ax4.set_xticklabels(gpt2_models, fontsize=10)
ax4.set_ylim(0, 10000)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)
fig4.tight_layout()
fig4.savefig('figures/fig_phase3_multimodel_placement.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_multimodel_placement.png")

# ============================================================
# Figure 5: Architecture Scaling — Hybrid Speedup + PP Memory
# ============================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left panel: speedup bars
colors5 = [CLR_GPT2, CLR_GPT2, CLR_T5]
bars5 = ax5a.bar(arch_models, arch_speedup, 0.5, color=colors5, edgecolor='black', linewidth=0.5)
for bar, sp in zip(bars5, arch_speedup):
    ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f'{sp}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax5a.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No speedup (1×)')
ax5a.set_ylabel('Hybrid Speedup (8-GPU / 1-GPU)', fontsize=11)
ax5a.set_title('Hybrid Parallelism Scaling by Architecture', fontsize=11)
ax5a.set_ylim(0, 3.2)
ax5a.legend(fontsize=8)
ax5a.grid(axis='y', alpha=0.3)

# Right panel: PP memory balance (stacked horizontal bars)
y_pos = np.arange(len(arch_models))
bars_s0 = ax5b.barh(y_pos, arch_mem_s0, 0.5, label='Stage 0', color='#6baed6',
                     edgecolor='black', linewidth=0.5)
bars_s1 = ax5b.barh(y_pos, [s1 - s0 for s0, s1 in zip(arch_mem_s0, arch_mem_s1)],
                     0.5, left=arch_mem_s0, label='Stage 1 (extra)',
                     color='#fc9272', edgecolor='black', linewidth=0.5)

for i in range(len(arch_models)):
    ax5b.text(arch_mem_s1[i] + 0.1, i, f'{arch_mem_s0[i]}/{arch_mem_s1[i]} GB',
              va='center', fontsize=9, fontweight='bold')

# Highlight T5 imbalance
ax5b.annotate('54% heavier →', xy=(3.87, 2), xytext=(4.5, 2.3),
              fontsize=9, fontweight='bold', color=CLR_T5,
              arrowprops=dict(arrowstyle='->', color=CLR_T5, lw=1.5))

ax5b.set_xlabel('Peak Memory per Stage (GB)', fontsize=11)
ax5b.set_title('Pipeline Stage Memory Balance', fontsize=11)
ax5b.set_yticks(y_pos)
ax5b.set_yticklabels(arch_models, fontsize=9)
ax5b.set_xlim(0, 6)
ax5b.legend(fontsize=8, loc='lower right')
ax5b.grid(axis='x', alpha=0.3)

fig5.tight_layout()
fig5.savefig('figures/fig_phase3_arch_scaling.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_arch_scaling.png")

# ============================================================
# Figure 6: T5 vs GPT-2 Single-Node Baselines
# ============================================================
fig6, ax6 = plt.subplots(figsize=(8, 5))
x6 = np.arange(len(baseline_configs))
w6 = 0.25

bars_t5b  = ax6.bar(x6 - w6, t5_baselines, w6, label='T5-base (237M)',
                     color=CLR_T5, edgecolor='black', linewidth=0.5)
bars_gm   = ax6.bar(x6, gpt2m_baselines, w6, label='GPT-2 Medium (354M)',
                     color=CLR_GPT2, edgecolor='black', linewidth=0.5)
bars_gs   = ax6.bar(x6 + w6, gpt2s_baselines, w6, label='GPT-2 Small (117M)',
                     color='#17becf', edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_t5b, t5_baselines):
    if val > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar, val in zip(bars_gm, gpt2m_baselines):
    if val > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar, val in zip(bars_gs, gpt2s_baselines):
    if val > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{val:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Mark missing PP data
ax6.text(x6[3], 300, 'PP(4)', ha='center', fontsize=7, color='gray', fontstyle='italic')

ax6.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax6.set_title('Single-Node Baselines: T5 vs GPT-2 (4 GPUs, PCIe)', fontsize=12)
ax6.set_xticks(x6)
ax6.set_xticklabels(baseline_configs, fontsize=10)
ax6.set_ylim(0, 9800)
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3)
fig6.tight_layout()
fig6.savefig('figures/fig_phase3_t5_baselines.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_phase3_t5_baselines.png")

# ============================================================
print("\n=== All Phase 3 figures generated ===")
print("  1. fig_phase3_pcie_vs_tcp.png        — PCIe vs TCP speedup")
print("  2. fig_phase3_good_vs_bad.png        — Good vs Bad placement")
print("  3. fig_phase3_three_way_placement.png — Three-way placement (GPT-2 + T5)")
print("  4. fig_phase3_multimodel_placement.png — Multi-model GPT-2 placement")
print("  5. fig_phase3_arch_scaling.png        — Architecture scaling + PP memory")
print("  6. fig_phase3_t5_baselines.png        — T5 vs GPT-2 baselines")
