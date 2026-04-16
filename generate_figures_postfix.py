"""
Generate figures for Phase 3 Post-Bugfix results.

BUG FIXED: configure() now creates TP-sharded pipeline stages when both
PP and TP are active, enabling true 3D parallelism (DP × PP × TP).

Run this after collecting results from run_phase3_postfix.sh.
Update the DATA section below with actual experiment numbers.

Usage:
    python generate_figures_postfix.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# ============================================================
# DATA — Update these after running experiments!
# ============================================================

# Single-GPU baselines (unchanged, from phase3)
single_gpu = {
    'GPT-2 Small':  3879,
    'GPT-2 Medium': 1404,
    'T5-base':      3359,
}

# Pre-fix results (from phase3 — TP was dead, effectively DP2×PP2)
prefix = {
    'good':  3519,  # GPT-2 Medium, dp=2,pp=2,tp=2(dead), good placement
    'bad':   2926,  # GPT-2 Medium, dp=2,pp=2,tp=2(dead), bad placement
    'worst': 3543,  # GPT-2 Medium, dp=2,pp=2,tp=2(dead), TP inter-node
}

# ========================================
# POST-FIX RESULTS — FILL IN AFTER EXPERIMENTS
# ========================================

# Group B: Good placement, GPT-2 Medium
postfix_good_bs8  = 0    # B1_3d_good_gpt2med_bs8 → tok/s
postfix_good_bs16 = 0    # B2_3d_good_gpt2med_bs16 → tok/s

# Group C: Bad placement, GPT-2 Medium
postfix_bad_bs8   = 0    # C1_3d_bad_gpt2med_bs8 → tok/s

# Group D: TP inter-node, GPT-2 Medium
postfix_worst_bs8 = 0    # D1_3d_worst_gpt2med_bs8 → tok/s

# Group E: Architecture comparison (good placement, bs=8)
postfix_arch = {
    'GPT-2 Small':  0,   # E1_3d_good_gpt2small → tok/s
    'GPT-2 Medium': 0,   # E2_3d_good_gpt2med → tok/s
    'T5-base':      0,   # E3_3d_good_t5base → tok/s
}

# Group E: Memory per stage (read from experiment logs)
postfix_mem = {
    'GPT-2 Small':  (0, 0),   # (stage0_GB, stage1_GB)
    'GPT-2 Medium': (0, 0),
    'T5-base':      (0, 0),
}

# Group F: ZeRO-1
postfix_zero1_bs8 = 0    # F1_3d_good_gpt2med_zero1 → tok/s

# ============================================================
# Style constants
# ============================================================
CLR_PREFIX  = '#999999'
CLR_POSTFIX = '#1f77b4'
CLR_GOOD    = '#2ca02c'
CLR_BAD     = '#d62728'
CLR_TP_INTER = '#ff7f0e'
CLR_GPT2    = '#1f77b4'
CLR_T5      = '#d62728'

# ============================================================
# Figure 1: Pre-Fix vs Post-Fix — Three-Way Placement (GPT-2 Medium)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(9, 5))
placements = ['Good\n(PP inter-node)', 'TP\ninter-node', 'Bad\n(DP inter-node)']
prefix_vals = [prefix['good'], prefix['worst'], prefix['bad']]
postfix_vals = [postfix_good_bs8, postfix_worst_bs8, postfix_bad_bs8]

x = np.arange(len(placements))
w = 0.35

bars_pre  = ax1.bar(x - w/2, prefix_vals, w, label='Pre-Fix (TP inactive)',
                     color=CLR_PREFIX, edgecolor='black', linewidth=0.5)
bars_post = ax1.bar(x + w/2, postfix_vals, w, label='Post-Fix (True 3D)',
                     color=CLR_POSTFIX, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_pre, prefix_vals):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars_post, postfix_vals):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.axhline(y=single_gpu['GPT-2 Medium'], color='gray', linestyle='--', linewidth=1,
            label=f"Single GPU ({single_gpu['GPT-2 Medium']:,} tok/s)", alpha=0.7)
ax1.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax1.set_title('Pre-Fix vs Post-Fix: Placement Comparison — GPT-2 Medium (8 GPUs)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(placements, fontsize=10)
ymax = max(max(prefix_vals), max(postfix_vals) if max(postfix_vals) > 0 else 4000) * 1.25
ax1.set_ylim(0, ymax)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('figures/fig_postfix_placement_comparison.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_postfix_placement_comparison.png")

# ============================================================
# Figure 2: Architecture Scaling — Pre-Fix vs Post-Fix Speedup
# ============================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

models = ['GPT-2 Small\n(117M)', 'GPT-2 Medium\n(354M)', 'T5-base\n(237M)']
model_keys = ['GPT-2 Small', 'GPT-2 Medium', 'T5-base']

# Pre-fix speedups (from phase3 results)
prefix_arch = [8678, 3519, 5790]
prefix_speedup = [v / single_gpu[k] for v, k in zip(prefix_arch, model_keys)]

# Post-fix speedups
postfix_arch_vals = [postfix_arch[k] for k in model_keys]
postfix_speedup = [v / single_gpu[k] if v > 0 else 0 for v, k in zip(postfix_arch_vals, model_keys)]

# Left: absolute throughput
x2 = np.arange(len(models))
w2 = 0.3
bars_pre2  = ax2a.bar(x2 - w2/2, prefix_arch, w2, label='Pre-Fix (TP inactive)',
                       color=CLR_PREFIX, edgecolor='black', linewidth=0.5)
bars_post2 = ax2a.bar(x2 + w2/2, postfix_arch_vals, w2, label='Post-Fix (True 3D)',
                       color=CLR_POSTFIX, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars_pre2, prefix_arch):
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
              f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars_post2, postfix_arch_vals):
    if val > 0:
        ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                  f'{val:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2a.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax2a.set_title('Throughput: Pre-Fix vs Post-Fix', fontsize=11)
ax2a.set_xticks(x2)
ax2a.set_xticklabels(models, fontsize=9)
ax2a.legend(fontsize=8)
ax2a.grid(axis='y', alpha=0.3)

# Right: speedup bars
colors = [CLR_GPT2, CLR_GPT2, CLR_T5]
bars_sp_pre = ax2b.bar(x2 - w2/2, prefix_speedup, w2, label='Pre-Fix',
                        color=CLR_PREFIX, edgecolor='black', linewidth=0.5)
bars_sp_post = ax2b.bar(x2 + w2/2, postfix_speedup, w2, label='Post-Fix',
                         color=CLR_POSTFIX, edgecolor='black', linewidth=0.5)

for bar, sp in zip(bars_sp_pre, prefix_speedup):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
              f'{sp:.2f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, sp in zip(bars_sp_post, postfix_speedup):
    if sp > 0:
        ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                  f'{sp:.2f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2b.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='1× (no speedup)')
ax2b.set_ylabel('Hybrid Speedup (8-GPU / 1-GPU)', fontsize=11)
ax2b.set_title('Scaling Efficiency: Pre-Fix vs Post-Fix', fontsize=11)
ax2b.set_xticks(x2)
ax2b.set_xticklabels(models, fontsize=9)
ax2b.set_ylim(0, max(max(prefix_speedup), max(postfix_speedup) if max(postfix_speedup) > 0 else 3) * 1.3)
ax2b.legend(fontsize=8)
ax2b.grid(axis='y', alpha=0.3)

fig2.tight_layout()
fig2.savefig('figures/fig_postfix_arch_scaling.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_postfix_arch_scaling.png")

# ============================================================
# Figure 3: Batch Size Impact (Good Placement, GPT-2 Medium)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(6, 4.5))
batch_labels = ['Pre-Fix\n(bs=8, TP dead)', 'Post-Fix\n(bs=8)', 'Post-Fix\n(bs=16)']
batch_vals = [prefix['good'], postfix_good_bs8, postfix_good_bs16]
bar_colors = [CLR_PREFIX, CLR_POSTFIX, CLR_GOOD]

bars3 = ax3.bar(batch_labels, batch_vals, 0.5, color=bar_colors,
                edgecolor='black', linewidth=0.5)

for bar, val in zip(bars3, batch_vals):
    if val > 0:
        sp = val / single_gpu['GPT-2 Medium']
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}\n({sp:.2f}×)', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

ax3.axhline(y=single_gpu['GPT-2 Medium'], color='gray', linestyle='--', linewidth=1,
            label=f"Single GPU ({single_gpu['GPT-2 Medium']:,} tok/s)", alpha=0.7)
ax3.set_ylabel('Aggregate Throughput (tokens/s)', fontsize=11)
ax3.set_title('Effect of Bugfix + Batch Size — GPT-2 Medium (8 GPUs, Good Placement)', fontsize=11)
ymax3 = max(batch_vals) * 1.3 if max(batch_vals) > 0 else 5000
ax3.set_ylim(0, ymax3)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig('figures/fig_postfix_batch_size.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_postfix_batch_size.png")

# ============================================================
print("\n=== Post-Bugfix figures generated ===")
print("  1. fig_postfix_placement_comparison.png — Pre-Fix vs Post-Fix placement")
print("  2. fig_postfix_arch_scaling.png         — Architecture scaling comparison")
print("  3. fig_postfix_batch_size.png           — Batch size impact")
print("\nNOTE: Update the DATA section with actual experiment numbers!")
