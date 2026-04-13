#!/usr/bin/env python3
"""Generate the corrected 3D mesh diagram showing GOOD placement (PP inter-node)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def generate_3d_mesh_good_placement():
    """
    Good placement: Mesh(pp=2, dp=2, tp=2)
    
    Rank | Coord (pp,dp,tp) | Node
    0    | (0, 0, 0)        | 0
    1    | (0, 0, 1)        | 0
    2    | (0, 1, 0)        | 0
    3    | (0, 1, 1)        | 0
    4    | (1, 0, 0)        | 1
    5    | (1, 0, 1)        | 1
    6    | (1, 1, 0)        | 1
    7    | (1, 1, 1)        | 1
    
    Groups:
      PP (axis 0, inter-node): {0,4}, {1,5}, {2,6}, {3,7}
      DP (axis 1, intra-node): {0,2}, {1,3}, {4,6}, {5,7}
      TP (axis 2, intra-node): {0,1}, {2,3}, {4,5}, {6,7}

    Layout: Two boxes = PP=0 (Node 0) and PP=1 (Node 1).
    Within each box: rows = DP, cols = TP.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    gpu_w, gpu_h = 1.4, 0.9

    # PP=0 slice (Node 0): ranks 0,1,2,3
    # Within: rows=DP (0 top, 1 bottom), cols=TP (0 left, 1 right)
    pp0_positions = {
        0: (1.5, 5.0),   # dp=0, tp=0
        1: (4.0, 5.0),   # dp=0, tp=1
        2: (1.5, 2.5),   # dp=1, tp=0
        3: (4.0, 2.5),   # dp=1, tp=1
    }
    # PP=1 slice (Node 1): ranks 4,5,6,7
    pp1_positions = {
        4: (7.5, 5.0),   # dp=0, tp=0
        5: (10.0, 5.0),  # dp=0, tp=1
        6: (7.5, 2.5),   # dp=1, tp=0
        7: (10.0, 2.5),  # dp=1, tp=1
    }
    all_pos = {**pp0_positions, **pp1_positions}

    # Background boxes for PP slices (= physical nodes)
    pp0_bg = FancyBboxPatch((0.2, 1.5), 5.2, 5.2, boxstyle="round,pad=0.3",
                             facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2, alpha=0.5)
    pp1_bg = FancyBboxPatch((6.2, 1.5), 5.2, 5.2, boxstyle="round,pad=0.3",
                             facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2, alpha=0.5)
    ax.add_patch(pp0_bg)
    ax.add_patch(pp1_bg)
    ax.text(2.75, 7.0, 'PP = 0  (Node 0)', fontsize=13, ha='center', fontweight='bold', color='#1565C0')
    ax.text(8.75, 7.0, 'PP = 1  (Node 1)', fontsize=13, ha='center', fontweight='bold', color='#E65100')

    # Draw GPU boxes
    for gpu_id, (x, y) in all_pos.items():
        rect = FancyBboxPatch((x - gpu_w/2, y - gpu_h/2), gpu_w, gpu_h,
                               boxstyle="round,pad=0.08",
                               facecolor='white', edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, f'GPU {gpu_id}', fontsize=10, ha='center', va='center', fontweight='bold')

    # TP edges (blue solid, horizontal within each row): same dp, same pp
    # TP groups: {0,1}, {2,3}, {4,5}, {6,7}
    tp_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for a, b in tp_pairs:
        y_off = 0.18
        ax.annotate('', xy=(all_pos[b][0] - gpu_w/2, all_pos[b][1] + y_off),
                    xytext=(all_pos[a][0] + gpu_w/2, all_pos[a][1] + y_off),
                    arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=2.5))

    # DP edges (green dotted, vertical within each box): same pp, same tp
    # DP groups: {0,2}, {1,3}, {4,6}, {5,7}  — ALL INTRA-NODE
    dp_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    for a, b in dp_pairs:
        x_off = -0.2
        ax.annotate('', xy=(all_pos[b][0] + x_off, all_pos[b][1] + gpu_h/2),
                    xytext=(all_pos[a][0] + x_off, all_pos[a][1] - gpu_h/2),
                    arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2.5, linestyle='dotted'))

    # PP edges (red dashed, horizontal between boxes): same dp, same tp
    # PP groups: {0,4}, {1,5}, {2,6}, {3,7}  — ALL INTER-NODE
    pp_pairs_offsets = [(0, 4, -0.08), (1, 5, -0.28), (2, 6, -0.08), (3, 7, -0.28)]
    for a, b, y_off in pp_pairs_offsets:
        ax.annotate('', xy=(all_pos[b][0] - gpu_w/2, all_pos[b][1] + y_off),
                    xytext=(all_pos[a][0] + gpu_w/2, all_pos[a][1] + y_off),
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2.5, linestyle='dashed'))

    # Axis labels
    ax.text(2.75, 5.7, 'TP', fontsize=10, ha='center', color='#1565C0', fontweight='bold')
    ax.text(0.75, 3.75, 'DP', fontsize=10, ha='center', color='#2E7D32', fontweight='bold', rotation=90)
    ax.text(5.75, 4.6, 'PP', fontsize=10, ha='center', color='#C62828', fontweight='bold')

    # Legend
    tp_l = mlines.Line2D([], [], color='#1565C0', lw=2.5, linestyle='-',
                          label='TP axis (all-reduce every layer)')
    dp_l = mlines.Line2D([], [], color='#2E7D32', lw=2.5, linestyle=':',
                          label='DP axis (all-reduce gradients)')
    pp_l = mlines.Line2D([], [], color='#C62828', lw=2.5, linestyle='--',
                          label='PP axis (send/recv activations)')
    ax.legend(handles=[tp_l, dp_l, pp_l], loc='lower center', fontsize=10,
              bbox_to_anchor=(0.5, -0.08), ncol=3, framealpha=0.9)

    # Plugin API text
    ax.text(5.75, 0.6, 'Plugin API:  plugin(tp=2, pp=2, zero=1)  →  dp = world_size / (tp × pp) = 2',
            fontsize=10, ha='center', va='center', style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1))

    plt.tight_layout()
    plt.savefig('figures/fig_ppt_3d_mesh_v3.png')
    plt.close()
    print('Saved fig_ppt_3d_mesh_v3.png')


if __name__ == '__main__':
    generate_3d_mesh_good_placement()
