#!/usr/bin/env python3
"""Generate images for the final presentation slides."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


# ================================================================
# IMAGE 1: Unified 3D Hybrid Plugin — Process Group Mesh
# ================================================================
def generate_3d_mesh():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Two 2x2 grids side by side representing DP=0 and DP=1 slices
    # Left grid: DP=0, Right grid: DP=1
    # Within each grid: rows=PP, cols=TP

    gpu_w, gpu_h = 1.4, 0.9

    # DP=0 slice positions
    dp0_positions = {
        0: (1.5, 5.0),   # pp=0, tp=0
        1: (4.0, 5.0),   # pp=0, tp=1
        2: (1.5, 2.5),   # pp=1, tp=0
        3: (4.0, 2.5),   # pp=1, tp=1
    }
    # DP=1 slice positions
    dp1_positions = {
        4: (7.5, 5.0),   # pp=0, tp=0
        5: (10.0, 5.0),  # pp=0, tp=1
        6: (7.5, 2.5),   # pp=1, tp=0
        7: (10.0, 2.5),  # pp=1, tp=1
    }
    all_pos = {**dp0_positions, **dp1_positions}

    # Background boxes for DP slices
    dp0_bg = FancyBboxPatch((0.2, 1.5), 5.2, 5.2, boxstyle="round,pad=0.3",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2, alpha=0.5)
    dp1_bg = FancyBboxPatch((6.2, 1.5), 5.2, 5.2, boxstyle="round,pad=0.3",
                             facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2, alpha=0.5)
    ax.add_patch(dp0_bg)
    ax.add_patch(dp1_bg)
    ax.text(2.75, 7.0, 'DP = 0', fontsize=13, ha='center', fontweight='bold', color='#2E7D32')
    ax.text(8.75, 7.0, 'DP = 1', fontsize=13, ha='center', fontweight='bold', color='#2E7D32')

    # Draw GPU boxes
    for gpu_id, (x, y) in all_pos.items():
        rect = FancyBboxPatch((x - gpu_w/2, y - gpu_h/2), gpu_w, gpu_h,
                               boxstyle="round,pad=0.08",
                               facecolor='white', edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, f'GPU {gpu_id}', fontsize=10, ha='center', va='center', fontweight='bold')

    # TP edges (blue solid, horizontal): same dp, same pp
    tp_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for a, b in tp_pairs:
        y_off = 0.18
        ax.annotate('', xy=(all_pos[b][0] - gpu_w/2, all_pos[b][1] + y_off),
                    xytext=(all_pos[a][0] + gpu_w/2, all_pos[a][1] + y_off),
                    arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=2.5))

    # PP edges (red dashed, vertical): same dp, same tp
    pp_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for a, b in pp_pairs:
        x_off = 0.2
        ax.annotate('', xy=(all_pos[b][0] + x_off, all_pos[b][1] + gpu_h/2),
                    xytext=(all_pos[a][0] + x_off, all_pos[a][1] - gpu_h/2),
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2.5, linestyle='dashed'))

    # DP edges (green dotted, horizontal between slices): same pp, same tp
    # Use different y-offsets so each of the 4 DP groups is visually distinct
    dp_pairs_offsets = [(0, 4, -0.08), (1, 5, -0.28), (2, 6, -0.08), (3, 7, -0.28)]
    for a, b, y_off in dp_pairs_offsets:
        ax.annotate('', xy=(all_pos[b][0] - gpu_w/2, all_pos[b][1] + y_off),
                    xytext=(all_pos[a][0] + gpu_w/2, all_pos[a][1] + y_off),
                    arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2.5, linestyle='dotted'))

    # Axis labels along edges
    ax.text(2.75, 5.7, 'TP', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
    ax.text(0.55, 3.75, 'PP', fontsize=10, ha='center', color='#C62828', fontweight='bold', rotation=90)
    ax.text(5.75, 4.6, 'DP', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')

    # Legend
    tp_l = mlines.Line2D([], [], color='#1565C0', lw=2.5, linestyle='-',
                          label='TP axis (all-reduce every layer)')
    pp_l = mlines.Line2D([], [], color='#C62828', lw=2.5, linestyle='--',
                          label='PP axis (send/recv activations)')
    dp_l = mlines.Line2D([], [], color='#2E7D32', lw=2.5, linestyle=':',
                          label='DP axis (all-reduce gradients)')
    ax.legend(handles=[tp_l, pp_l, dp_l], loc='lower center', fontsize=10,
              bbox_to_anchor=(0.5, -0.08), ncol=3, framealpha=0.9)

    # Plugin API text
    ax.text(5.75, 0.6, 'Plugin API:  plugin(tp=2, pp=2, zero=1)  →  dp = world_size / (tp × pp) = 2',
            fontsize=10, ha='center', va='center', style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1))

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_3d_mesh.png')
    plt.close()
    print('Saved fig_ppt_3d_mesh.png')


# ================================================================
# IMAGE 2: Communication-Aware Placement (Clear, no overlapping)
# ================================================================
def generate_comm_aware_placement():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1.5, 15.5)
    ax.set_ylim(-1.5, 9.0)
    ax.set_aspect('equal')
    ax.axis('off')

    gpu_w, gpu_h = 1.5, 0.9

    # Node 0 box
    n0 = FancyBboxPatch((-0.3, 0.8), 6.1, 7.0, boxstyle="round,pad=0.3",
                          facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2.5)
    # Node 1 box
    n1 = FancyBboxPatch((8.2, 0.8), 6.1, 7.0, boxstyle="round,pad=0.3",
                          facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.5)
    ax.add_patch(n0)
    ax.add_patch(n1)

    ax.text(2.75, 8.2, 'Node 0', fontsize=14, ha='center', fontweight='bold', color='#1565C0')
    ax.text(2.75, 7.7, 'PCIe ~15 GB/s', fontsize=10, ha='center', color='#1565C0', style='italic')
    ax.text(11.25, 8.2, 'Node 1', fontsize=14, ha='center', fontweight='bold', color='#E65100')
    ax.text(11.25, 7.7, 'PCIe ~15 GB/s', fontsize=10, ha='center', color='#E65100', style='italic')

    # GPU positions: 2x2 within each node, well spaced
    pos = {
        0: (1.2, 6.0),  1: (4.3, 6.0),
        2: (1.2, 2.5),  3: (4.3, 2.5),
        4: (9.7, 6.0),  5: (12.8, 6.0),
        6: (9.7, 2.5),  7: (12.8, 2.5),
    }

    for gid, (x, y) in pos.items():
        rect = FancyBboxPatch((x - gpu_w/2, y - gpu_h/2), gpu_w, gpu_h,
                               boxstyle="round,pad=0.08",
                               facecolor='white', edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, f'GPU {gid}', fontsize=11, ha='center', va='center', fontweight='bold')

    # --- TP links (blue solid, HORIZONTAL within each row) ---
    tp_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for a, b in tp_pairs:
        y = pos[a][1] + 0.2  # slightly above center
        ax.annotate('', xy=(pos[b][0] - gpu_w/2 - 0.05, y),
                    xytext=(pos[a][0] + gpu_w/2 + 0.05, y),
                    arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=3))
        mid_x = (pos[a][0] + pos[b][0]) / 2
        ax.text(mid_x, y + 0.35, 'TP', fontsize=9, ha='center', color='#1565C0', fontweight='bold')

    # --- DP links (green dotted, VERTICAL within each column) ---
    dp_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for a, b in dp_pairs:
        x = pos[a][0] + 0.25  # slightly right of center
        ax.annotate('', xy=(x, pos[b][1] + gpu_h/2 + 0.05),
                    xytext=(x, pos[a][1] - gpu_h/2 - 0.05),
                    arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=3, linestyle=(0, (2, 2))))
        mid_y = (pos[a][1] + pos[b][1]) / 2
        ax.text(x + 0.4, mid_y, 'DP', fontsize=9, ha='center', color='#2E7D32',
                fontweight='bold', rotation=90)

    # --- PP links (red dashed, HORIZONTAL between nodes) ---
    # Use different y-offsets so all 4 PP groups are visually distinct
    pp_pairs_offsets = [(0, 4, -0.10), (1, 5, -0.35), (2, 6, -0.10), (3, 7, -0.35)]
    for a, b, y_off in pp_pairs_offsets:
        y = pos[a][1] + y_off
        ax.annotate('', xy=(pos[b][0] - gpu_w/2 - 0.05, y),
                    xytext=(pos[a][0] + gpu_w/2 + 0.05, y),
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=3, linestyle='dashed'))

    # PP labels in the gap
    ax.text(7.0, 6.2, 'PP', fontsize=11, ha='center', color='#C62828', fontweight='bold')
    ax.text(7.0, 2.8, 'PP', fontsize=11, ha='center', color='#C62828', fontweight='bold')
    ax.text(7.0, 4.5, 'TCP ~0.6 GB/s', fontsize=10, ha='center', color='#C62828', style='italic')

    # Legend at bottom
    tp_l = mlines.Line2D([], [], color='#1565C0', lw=3, linestyle='-',
                          label='TP — intra-node (PCIe)')
    dp_l = mlines.Line2D([], [], color='#2E7D32', lw=3, linestyle=':',
                          label='DP — intra-node (PCIe)')
    pp_l = mlines.Line2D([], [], color='#C62828', lw=3, linestyle='--',
                          label='PP — inter-node (TCP)')
    ax.legend(handles=[tp_l, dp_l, pp_l], loc='lower center', fontsize=11,
              bbox_to_anchor=(0.5, -0.10), ncol=3, framealpha=0.9)

    ax.set_title('"Good" Placement: Heaviest communication stays on fastest link',
                 fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_comm_aware_placement.png')
    plt.close()
    print('Saved fig_ppt_comm_aware_placement.png')


# ================================================================
# IMAGE 3: GPT-2 Medium — Three Placement Strategies
# ================================================================
def generate_gpt2_three_way():
    placements = ['Good\n(PP inter-node)', 'TP inter-node', 'Bad\n(DP inter-node)']
    throughputs = [3519, 3543, 2926]
    colors = ['#4CAF50', '#FF9800', '#E53935']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(placements, throughputs, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,} tok/s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Percentage annotation between Good and Bad
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


# ================================================================
# IMAGE 4: T5-base — Three Placement Strategies
# ================================================================
def generate_t5_three_way():
    placements = ['Good\n(PP inter-node)', 'TP inter-node', 'Bad\n(DP inter-node)']
    throughputs = [5790, 5838, 4558]
    colors = ['#4CAF50', '#FF9800', '#E53935']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(placements, throughputs, color=colors, width=0.5,
                  edgecolor='black', linewidth=0.8)

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


# ================================================================
# IMAGE 5: Multi-Model Scaling — Single GPU vs 8-GPU Hybrid
# ================================================================
def generate_model_scaling():
    models = ['GPT-2 Small\n(117M)', 'GPT-2 Medium\n(354M)', 'T5-base\n(237M)']
    single_gpu = [3879, 1404, 3359]
    hybrid_8gpu = [8678, 3519, 5790]
    speedups = [2.24, 2.51, 1.72]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, single_gpu, width, label='Single GPU', color='#90CAF9',
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, hybrid_8gpu, width, label='8-GPU Hybrid (Good placement)',
                   color='#1565C0', edgecolor='black', linewidth=0.8)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, color='#555')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, color='#333',
                fontweight='bold')

    # Speedup annotations
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

    # Highlight T5 is mid-sized but scales worst
    ax.annotate('Smaller model, but scales better\nthan T5 → architectural effect',
                xy=(0 + width/2, hybrid_8gpu[0] + 350),
                xytext=(1.0, 7500),
                fontsize=9, ha='center', color='#555', style='italic',
                arrowprops=dict(arrowstyle='->', color='#999', lw=1))

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_model_scaling.png')
    plt.close()
    print('Saved fig_ppt_model_scaling.png')


# ================================================================
# IMAGE 6: How row-major axis ordering determines placement
# ================================================================
def generate_axis_placement_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-2.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ---- TITLE ----
    ax.text(6.0, 9.2, 'How Mesh(axis0, axis1, axis2) Maps Ranks to Nodes',
            fontsize=16, ha='center', fontweight='bold', color='#333')

    # ---- KEY RULE (highlighted box) ----
    rule_box = FancyBboxPatch((0.5, 7.8), 11.0, 1.0, boxstyle='round,pad=0.2',
                               facecolor='#FFF9C4', edgecolor='#F9A825', lw=2)
    ax.add_patch(rule_box)
    ax.text(6.0, 8.5, 'Row-major order:  axis 0 changes slowest  →  spans across nodes (slow TCP)',
            fontsize=12, ha='center', va='center', color='#C62828', fontweight='bold')
    ax.text(6.0, 8.05, 'axis 2 changes fastest  →  stays within a node (fast PCIe)',
            fontsize=11, ha='center', va='center', color='#1565C0', fontweight='bold')

    # ---- EXAMPLE: Good placement Mesh(pp=2, dp=2, tp=2) ----
    ax.text(6.0, 7.3, 'Example:  Mesh(pp=2, dp=2, tp=2)     →  axis 0 = PP,  axis 1 = DP,  axis 2 = TP',
            fontsize=11, ha='center', color='#333', family='monospace')

    # Rank table: show how row-major assigns coordinates
    # Columns: Rank | coord (a0,a1,a2) | PP rank | DP rank | TP rank | Node
    coords = {}
    for r in range(8):
        a2 = r % 2       # TP (fastest)
        a1 = (r // 2) % 2  # DP
        a0 = r // 4        # PP (slowest)
        coords[r] = (a0, a1, a2)

    # Draw the rank-to-coordinate table
    table_top = 6.8
    col_x = [1.0, 2.8, 4.8, 6.3, 7.8, 9.5]
    headers = ['Rank', 'Coord\n(a0,a1,a2)', 'PP\n(axis 0)', 'DP\n(axis 1)', 'TP\n(axis 2)', 'Node']
    header_colors = ['#333', '#333', '#C62828', '#2E7D32', '#1565C0', '#333']

    for x, h, c in zip(col_x, headers, header_colors):
        ax.text(x, table_top, h, fontsize=9, ha='center', va='center',
                fontweight='bold', color=c)

    # Separator line
    ax.plot([0.2, 10.3], [table_top - 0.35, table_top - 0.35], color='#999', lw=0.8)

    for r in range(8):
        a0, a1, a2 = coords[r]
        y = table_top - 0.7 - r * 0.45
        node = 0 if a0 == 0 else 1
        bg_color = '#E3F2FD' if node == 0 else '#FFF3E0'
        # Row background
        bg = FancyBboxPatch((0.2, y - 0.18), 10.1, 0.38, boxstyle='round,pad=0.02',
                             facecolor=bg_color, edgecolor='none', alpha=0.5)
        ax.add_patch(bg)

        vals = [f'R{r}', f'({a0}, {a1}, {a2})', str(a0), str(a1), str(a2), f'Node {node}']
        val_colors = ['#333', '#555', '#C62828', '#2E7D32', '#1565C0', '#333']
        for x, v, c in zip(col_x, vals, val_colors):
            fw = 'bold' if x == col_x[0] or x == col_x[-1] else 'normal'
            ax.text(x, y, v, fontsize=9, ha='center', va='center', color=c, fontweight=fw)

    # ---- RIGHT SIDE: Visual node layout ----
    # Node 0 box
    n0 = FancyBboxPatch((10.6, 4.5), 1.1, 2.2, boxstyle='round,pad=0.1',
                          facecolor='#E3F2FD', edgecolor='#1565C0', lw=2)
    ax.add_patch(n0)
    ax.text(11.15, 6.85, 'Node 0', fontsize=10, ha='center', fontweight='bold', color='#1565C0')
    for i, r in enumerate([0, 1, 2, 3]):
        row, col = divmod(i, 2)
        x, y = 10.85 + col * 0.6, 6.2 - row * 0.7
        rect = FancyBboxPatch((x - 0.22, y - 0.2), 0.44, 0.4, boxstyle='round,pad=0.03',
                               facecolor='white', edgecolor='#333', lw=1)
        ax.add_patch(rect)
        ax.text(x, y, f'R{r}', fontsize=8, ha='center', va='center', fontweight='bold')

    # Node 1 box
    n1 = FancyBboxPatch((10.6, 1.5), 1.1, 2.2, boxstyle='round,pad=0.1',
                          facecolor='#FFF3E0', edgecolor='#E65100', lw=2)
    ax.add_patch(n1)
    ax.text(11.15, 3.85, 'Node 1', fontsize=10, ha='center', fontweight='bold', color='#E65100')
    for i, r in enumerate([4, 5, 6, 7]):
        row, col = divmod(i, 2)
        x, y = 10.85 + col * 0.6, 3.2 - row * 0.7
        rect = FancyBboxPatch((x - 0.22, y - 0.2), 0.44, 0.4, boxstyle='round,pad=0.03',
                               facecolor='white', edgecolor='#333', lw=1)
        ax.add_patch(rect)
        ax.text(x, y, f'R{r}', fontsize=8, ha='center', va='center', fontweight='bold')

    # Arrow between nodes: axis 0 = PP inter-node
    ax.annotate('', xy=(11.15, 3.75), xytext=(11.15, 4.5),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2.5, linestyle='dashed'))
    ax.text(12.0, 4.15, 'axis 0\n= PP\n(TCP)', fontsize=8, ha='center', va='center',
            color='#C62828', fontweight='bold')

    # Intra-node label
    ax.text(12.0, 5.6, 'axes 1,2\n= DP,TP\n(PCIe)', fontsize=7.5, ha='center', va='center',
            color='#1565C0', fontweight='bold')

    # ---- BOTTOM: Summary of three placements ----
    summary_y = -0.4
    ax.plot([0.2, 10.3], [summary_y + 0.5, summary_y + 0.5], color='#999', lw=0.8)
    ax.text(0.5, summary_y + 0.8, 'To switch placement, just change which axis is in position 0:',
            fontsize=10, color='#333', fontweight='bold')

    placements_text = [
        ('Good:', 'Mesh(PP, DP, TP)', 'PP inter-node  →  lightest traffic on slow link', '#4CAF50'),
        ('Bad:',  'Mesh(DP, PP, TP)', 'DP inter-node  →  heaviest traffic on slow link', '#C62828'),
        ('TP inter:', 'Mesh(TP, PP, DP)', 'TP inter-node  →  frequent but small messages on slow link', '#E65100'),
    ]
    for i, (label, code, desc, color) in enumerate(placements_text):
        y = summary_y - i * 0.55
        ax.text(0.7, y, label, fontsize=10, fontweight='bold', color=color, va='center')
        ax.text(2.5, y, code, fontsize=10, family='monospace', color='#333', va='center')
        ax.text(6.2, y, desc, fontsize=9.5, color='#555', va='center', style='italic')

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_axis_placement.png')
    plt.close()
    print('Saved fig_ppt_axis_placement.png')


# ================================================================
# Generate all
# ================================================================
if __name__ == '__main__':
    generate_3d_mesh()
    generate_comm_aware_placement()
    generate_gpt2_three_way()
    generate_t5_three_way()
    generate_model_scaling()
    generate_axis_placement_diagram()
    print('\nAll presentation images generated!')
