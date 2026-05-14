"""
draw_pipeline.py — Task 1.2
Vẽ pipeline diagram 4 module bằng matplotlib.
Output: outputs/figures/pipeline_diagram.png
"""

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_pipeline():
    out_dir = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'pipeline_diagram.png'

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    COLORS = {
        'input':  '#6c757d',
        'mod_a':  '#1f77b4',
        'mod_b':  '#ff7f0e',
        'mod_c':  '#2ca02c',
        'mod_d':  '#d62728',
        'arrow':  '#343a40',
    }

    def draw_box(ax, x, y, w, h, color, title, subtitle, text_color='white'):
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.1',
            linewidth=2,
            edgecolor='white',
            facecolor=color,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h * 0.65,
            title,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=text_color, zorder=4,
        )
        if subtitle:
            ax.text(
                x + w / 2, y + h * 0.35,
                subtitle,
                ha='center', va='center',
                fontsize=9,
                color=text_color,
                style='italic',
                zorder=4,
                wrap=True,
            )

    def draw_arrow(ax, x1, y1, x2, y2, label='', color='#343a40'):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=2,
            ),
            zorder=5,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.18
            ax.text(
                mx, my, label,
                ha='center', va='bottom',
                fontsize=8.5,
                color='#495057',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85),
                zorder=6,
            )

    def draw_image_icon(ax, x, y, size=1.0, color='#6c757d'):
        outer = mpatches.FancyBboxPatch(
            (x - size / 2, y - size / 2), size, size,
            boxstyle='round,pad=0.05',
            linewidth=1.5,
            edgecolor=color,
            facecolor='white',
            zorder=3,
        )
        ax.add_patch(outer)
        inner_h = size * 0.55
        ax.plot(
            [x - inner_h / 2, x + inner_h / 2],
            [y - inner_h / 2, y + inner_h / 2],
            color=color, lw=1.2, zorder=4,
        )
        ax.plot(
            [x - inner_h / 2, x + inner_h / 2],
            [y + inner_h / 2, y - inner_h / 2],
            color=color, lw=1.2, zorder=4,
        )

    # ── Title ──────────────────────────────────────────────
    ax.text(
        8, 8.4,
        'Face Recognition Pipeline — 4-Module Architecture',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color='#212529',
    )
    ax.text(
        8, 7.95,
        'Input Image  \u2192  ArcFace Embedding  \u2192  IQA  \u2192  Adaptive Threshold  \u2192  Gallery Decision',
        ha='center', va='center',
        fontsize=9,
        color='#6c757d',
    )

    # ── INPUT ──────────────────────────────────────────────
    in_x, in_y = 0.8, 4.5
    draw_image_icon(ax, in_x, in_y, size=1.1, color=COLORS['input'])
    ax.text(in_x, in_y - 0.95, 'Input Image\n(112\u00d7112)', ha='center', va='top',
            fontsize=8.5, color='#495057')

    # ── MODULE A ───────────────────────────────────────────
    mx_a, my_a = 4.2, 4.0
    mw, mh = 3.5, 2.8
    draw_box(
        ax, mx_a, my_a, mw, mh, COLORS['mod_a'],
        'Module A', 'ArcFace buffalo_sc\nembedding: 512-d',
    )
    ax.text(
        mx_a + mw / 2, my_a - 0.25,
        'InsightFace ONNX\nCPU-friendly',
        ha='center', va='top',
        fontsize=7.5, color='#6c757d',
    )

    # ── MODULE B ───────────────────────────────────────────
    mx_b, my_b = 8.0, 4.0
    draw_box(
        ax, mx_b, my_b, mw, mh, COLORS['mod_b'],
        'Module B', 'IQA\nL (luminance), N (noise)',
    )
    ax.text(
        mx_b + mw / 2, my_b - 0.25,
        'YCbCr + Gaussian Blur\nq = 1 \u2212 N',
        ha='center', va='top',
        fontsize=7.5, color='#6c757d',
    )

    # ── MODULE C ───────────────────────────────────────────
    mx_c, my_c = 11.8, 4.0
    draw_box(
        ax, mx_c, my_c, mw, mh, COLORS['mod_c'],
        'Module C', '\u03c4(C) — Adaptive Threshold\n\u03c4 = f(L, N, q, \u03b3)',
    )
    ax.text(
        mx_c + mw / 2, my_c - 0.25,
        '4 formulas:\nfixed | bin | linear | interaction',
        ha='center', va='top',
        fontsize=7.5, color='#6c757d',
    )

    # ── MODULE D ───────────────────────────────────────────
    mx_d, my_d = 15.0, 4.0
    mw_d = 2.8
    draw_box(
        ax, mx_d, my_d, mw_d, mh, COLORS['mod_d'],
        'Module D', 'Gallery Manager\n1:N search',
    )
    ax.text(
        mx_d + mw_d / 2, my_d - 0.25,
        'Accept / Reject\n+ online update',
        ha='center', va='top',
        fontsize=7.5, color='#6c757d',
    )

    # ── OUTPUT ────────────────────────────────────────────
    out_x, out_y = 15.0, 2.2
    for i, (label, color) in enumerate([
        ('Accept \u2713', '#2ca02c'),
        ('Reject \u2717', '#d62728'),
    ]):
        yy = out_y - i * 0.6
        ax.text(
            out_x, yy,
            '\u25b6',
            ha='center', va='center',
            fontsize=10,
            color=color,
            zorder=4,
        )
        ax.text(
            out_x + 0.3, yy,
            label,
            ha='left', va='center',
            fontsize=9,
            color=color,
            fontweight='bold',
            zorder=4,
        )
    ax.text(out_x + 0.1, out_y - 1.4, 'Decision', ha='center', va='top',
            fontsize=8, color='#6c757d')

    # ── ARROWS ────────────────────────────────────────────
    draw_arrow(ax, in_x + 0.55, in_y, mx_a, my_a + mh / 2,
               label='RGB image')
    draw_arrow(ax, mx_a + mw, my_a + mh / 2, mx_b, my_b + mh / 2,
               label='embedding (512-d)')
    draw_arrow(ax, mx_b + mw, my_b + mh / 2, mx_c, my_c + mh / 2,
               label='L, N, q, bin_id')
    draw_arrow(ax, mx_c + mw, my_c + mh / 2, mx_d, my_d + mh / 2,
               label='\u03c4 value')
    draw_arrow(ax, mx_d + mw_d / 2, my_d, out_x, out_y - 0.1,
               label='')
    draw_arrow(ax, mx_d + mw_d / 2, my_d, out_x, out_y + 0.5,
               label='')

    # ── LEGEND ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=COLORS['mod_a'], label='Module A: ArcFace Embedding'),
        mpatches.Patch(facecolor=COLORS['mod_b'], label='Module B: IQA'),
        mpatches.Patch(facecolor=COLORS['mod_c'], label='Module C: Adaptive Threshold'),
        mpatches.Patch(facecolor=COLORS['mod_d'], label='Module D: Gallery Manager'),
    ]
    ax.legend(
        handles=legend_items,
        loc='lower left',
        bbox_to_anchor=(0.0, 0.0),
        fontsize=8.5,
        framealpha=0.9,
        edgecolor='#dee2e6',
    )

    # ── DATA FLOW LABELS (right side) ─────────────────────
    for label, y_pos in [
        ('cosine_sim', my_a + mh / 2),
        ('tau_compare', my_c + mh / 2),
        ('best_match', my_d + mh / 2),
    ]:
        pass

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Pipeline diagram saved: {out_path}")


if __name__ == '__main__':
    draw_pipeline()
