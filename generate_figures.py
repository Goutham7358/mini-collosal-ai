"""Generate figures for Phase 2 report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data (cross-checked against project_phase_2.tex tables)
# ============================================================

# --- Figure 1: DP variants throughput progression ---
dp_methods = ['Naive', 'Ring', 'Ring\nbucketed', 'NCCL allreduce\nbucketed']
dp_throughput = [492, 1082, 1258, 1393]
dp_scaling = [9.0, 19.9, 23.1, 25.6]

# --- Figure 2: Throughput vs Memory scatter (all configs) ---
configs = {
    'Single GPU':          (1361, 7.67),
    'DP naive':            (492,  7.67),
    'DP ring':             (1082, 7.67),
    'DP ring bucket':      (1258, 8.14),
    'DP NCCL allred.':     (1393, 7.67),
    'ZeRO-1':              (934,  9.47),
    'ZeRO-2':              (915,  6.96),
    'TP 4-way':            (959,  3.38),
    'PP naive':            (3050, 3.09),
    'PP 1F1B':             (3070, 2.17),
    'DP(2)×TP(2)':         (2327, 4.80),
    'DP(2)×PP(2)':         (2281, 4.08),
    'PP(2)+ZeRO-1':        (1742, 5.74),
}

# --- Figure 3: Summary bar chart ---
summary_names = [
    'Single GPU', 'DP naive', 'DP ring', 'DP ring\nbucket',
    'DP NCCL\nallred.', 'ZeRO-1', 'ZeRO-2', 'TP 4-way',
    'PP naive', 'PP 1F1B', 'DP(2)\n×TP(2)', 'DP(2)\n×PP(2)',
    'PP(2)+\nZeRO-1'
]
summary_tput = [1361, 492, 1082, 1258, 1393, 934, 915, 959, 3050, 3070, 2327, 2281, 1742]
summary_mem  = [7.67, 7.67, 7.67, 8.14, 7.67, 9.47, 6.96, 3.38, 3.09, 2.17, 4.80, 4.08, 5.74]

# ============================================================
# Figure 1: DP Throughput Progression
# ============================================================
fig1, ax1 = plt.subplots(figsize=(7, 4))
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax1.bar(dp_methods, dp_throughput, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

# Add value labels on bars
for bar, tput, scale in zip(bars, dp_throughput, dp_scaling):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{tput} tok/s\n({scale}% eff.)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Single GPU reference line
ax1.axhline(y=1361, color='gray', linestyle='--', linewidth=1, label='Single GPU (1,361 tok/s)')
ax1.legend(fontsize=9, loc='upper left')

ax1.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax1.set_title('Data Parallelism: Effect of Gradient Sync Algorithm (4 GPUs)', fontsize=12)
ax1.set_ylim(0, 1700)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('figures/fig_dp_throughput.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_dp_throughput.png")

# ============================================================
# Figure 2: Throughput vs Memory Scatter
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5.5))

# Color groups
color_map = {
    'Single GPU': '#555555',
    'DP naive': '#d62728', 'DP ring': '#d62728',
    'DP ring bucket': '#d62728', 'DP NCCL allred.': '#d62728',
    'ZeRO-1': '#ff7f0e', 'ZeRO-2': '#ff7f0e',
    'TP 4-way': '#2ca02c',
    'PP naive': '#9467bd', 'PP 1F1B': '#9467bd',
    'DP(2)×TP(2)': '#1f77b4', 'DP(2)×PP(2)': '#1f77b4',
    'PP(2)+ZeRO-1': '#1f77b4',
}
marker_map = {
    'Single GPU': 's',
    'DP naive': 'o', 'DP ring': 'o', 'DP ring bucket': 'o', 'DP NCCL allred.': 'o',
    'ZeRO-1': '^', 'ZeRO-2': '^',
    'TP 4-way': 'D',
    'PP naive': 'v', 'PP 1F1B': 'v',
    'DP(2)×TP(2)': 'P', 'DP(2)×PP(2)': 'P', 'PP(2)+ZeRO-1': 'P',
}

for name, (tput, mem) in configs.items():
    ax2.scatter(mem, tput, c=color_map[name], marker=marker_map[name],
                s=100, edgecolors='black', linewidth=0.5, zorder=5)

# Label each point with offset to avoid overlap
offsets = {
    'Single GPU':      (10, -5),
    'DP naive':        (-10, -15),
    'DP ring':         (10, 5),
    'DP ring bucket':  (10, -15),
    'DP NCCL allred.': (10, 5),
    'ZeRO-1':          (10, -5),
    'ZeRO-2':          (10, -5),
    'TP 4-way':        (10, -5),
    'PP naive':        (10, 5),
    'PP 1F1B':         (10, -15),
    'DP(2)×TP(2)':     (10, 5),
    'DP(2)×PP(2)':     (-15, 10),
    'PP(2)+ZeRO-1':    (10, -5),
}

for name, (tput, mem) in configs.items():
    ox, oy = offsets[name]
    ax2.annotate(name, (mem, tput), textcoords="offset points",
                 xytext=(ox, oy), fontsize=7.5, ha='left')

# Legend for groups
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555', markersize=8, label='Baseline'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='Data Parallel'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#ff7f0e', markersize=8, label='ZeRO'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ca02c', markersize=8, label='Tensor Parallel'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#9467bd', markersize=8, label='Pipeline Parallel'),
    Line2D([0], [0], marker='P', color='w', markerfacecolor='#1f77b4', markersize=8, label='Hybrid'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='center right')

ax2.set_xlabel('Peak GPU Memory (GB)', fontsize=11)
ax2.set_ylabel('Throughput (tokens/s)', fontsize=11)
ax2.set_title('Throughput vs. Peak Memory: All Configurations', fontsize=12)
ax2.grid(alpha=0.3)
ax2.set_xlim(1.5, 10.5)
ax2.set_ylim(200, 3400)
fig2.tight_layout()
fig2.savefig('figures/fig_throughput_vs_memory.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_throughput_vs_memory.png")

# ============================================================
# Figure 3: Summary Bar Chart
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

x = np.arange(len(summary_names))
width = 0.6

# Throughput bars
bar_colors = (
    ['#555555'] +          # Single GPU
    ['#d62728'] * 4 +      # DP variants
    ['#ff7f0e'] * 2 +      # ZeRO
    ['#2ca02c'] +          # TP
    ['#9467bd'] * 2 +      # PP
    ['#1f77b4'] * 3        # Hybrid
)

bars3a = ax3a.bar(x, summary_tput, width, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(summary_tput):
    ax3a.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=7, fontweight='bold')
ax3a.set_ylabel('Throughput (tok/s)', fontsize=10)
ax3a.set_title('All Configurations: Throughput and Memory', fontsize=12)
ax3a.axhline(y=1361, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax3a.set_ylim(0, 3500)
ax3a.grid(axis='y', alpha=0.3)

# Memory bars
bars3b = ax3b.bar(x, summary_mem, width, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(summary_mem):
    ax3b.text(i, v + 0.15, f'{v}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax3b.set_ylabel('Peak Memory (GB)', fontsize=10)
ax3b.set_xticks(x)
ax3b.set_xticklabels(summary_names, fontsize=7.5, rotation=0)
ax3b.axhline(y=7.67, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax3b.set_ylim(0, 11)
ax3b.grid(axis='y', alpha=0.3)

fig3.tight_layout()
fig3.savefig('figures/fig_summary_all.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_summary_all.png")

# ============================================================
# Cross-check: print all values
# ============================================================
print("\n=== DATA CROSS-CHECK ===")
print("\nDP variants (Table II):")
for m, t, s in zip(dp_methods, dp_throughput, dp_scaling):
    print(f"  {m.replace(chr(10),' '):30s}  {t:5d} tok/s  {s}% eff")

print("\nAll configs (Table VII):")
for name, (tput, mem) in configs.items():
    print(f"  {name:20s}  {tput:5d} tok/s  {mem:.2f} GB")
