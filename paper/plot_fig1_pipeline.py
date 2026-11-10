#!/usr/bin/env python3
"""
生成 Paper 2 图 1：Echo-MF-2kD 的两阶段免训练平均场伴随乘积流形初始化与全角度量子能量重耦合架构图。
严格遵循 Nature Computational Science 出版规范（宽度 183 mm，纯矢量无重叠排版）。
"""

from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 6.8,
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
    'figure.facecolor': 'white'
})


def create_fig1():
    """绘制并导出 Paper 2 图 1 架构与物理机制总览图。"""
    mm_to_inch = 1.0 / 25.4
    fig_w, fig_h = 183.0 * mm_to_inch, 118.0 * mm_to_inch
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    C_NAVY = '#1E3A8A'
    C_TEAL = '#0F766E'
    C_AMBER = '#B45309'
    C_PURPLE = '#6D28D9'
    C_DARK = '#0F172A'
    C_MUTED = '#334155'
    C_BG_TOP = '#F8FAFC'
    C_BG_BOT = '#F0FDF4'
    BORDER_TOP = '#94A3B8'
    BORDER_BOT = '#6EE7B7'

    # Stage 1 Top Band
    rect_top = FancyBboxPatch(
        (1.0, 54.0), 98.0, 44.2,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        facecolor=C_BG_TOP, edgecolor=BORDER_TOP, linewidth=0.9, zorder=1
    )
    ax.add_patch(rect_top)
    ax.text(
        2.6, 95.6,
        'STAGE 1: Training-Free Classical 2kD Spin-Echo Product-Manifold Initialization (0 Quantum Evaluations)',
        fontsize=7.6, fontweight='bold', color=C_NAVY, zorder=2
    )

    # Stage 2 Bottom Band
    rect_bot = FancyBboxPatch(
        (1.0, 1.8), 98.0, 44.5,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        facecolor=C_BG_BOT, edgecolor=BORDER_BOT, linewidth=0.9, zorder=1
    )
    ax.add_patch(rect_bot)
    ax.text(
        14.2, 43.8,
        'STAGE 2: Full-Angle 2p-D Quantum Energy Recoupling (Unmodified Global Hardware-Native QAOA Circuit)',
        fontsize=7.2, fontweight='bold', color=C_TEAL, zorder=2
    )

    def draw_card(x, y, w, h, items, border='#94A3B8', bg='white', z=3):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=1.1",
            facecolor=bg, edgecolor=border, linewidth=0.85, zorder=z
        )
        ax.add_patch(box)
        n_items = len(items)
        pad_top = 2.9
        pad_bot = 2.5
        avail_h = h - pad_top - pad_bot
        step = avail_h / max(1, n_items - 0.15)
        y_start = y + h - pad_top - 0.38 * step

        for idx, item in enumerate(items):
            curr_y = y_start - idx * step
            itype = item.get('type', 'text')
            text = item.get('text', '')
            if itype == 'title':
                f_size = item.get('size', 6.8)
                ax.text(
                    x + w / 2, curr_y, text,
                    fontsize=f_size, fontweight='bold',
                    ha='center', va='center', color=C_DARK, zorder=z + 1
                )
            elif itype == 'subtitle':
                color = item.get('color', C_NAVY)
                f_size = item.get('size', 6.1)
                ax.text(
                    x + w / 2, curr_y, text,
                    fontsize=f_size, fontstyle='italic',
                    ha='center', va='center', color=color, zorder=z + 1
                )
            elif itype == 'badge':
                b_color = item.get('color', C_TEAL)
                b_w = item.get('w', w - 2.0)
                f_size = item.get('size', 5.65)
                bbox = FancyBboxPatch(
                    (x + w / 2 - b_w / 2, curr_y - 1.15), b_w, 2.3,
                    boxstyle="round,pad=0.08,rounding_size=0.5",
                    facecolor=b_color, edgecolor='none', zorder=z + 1
                )
                ax.add_patch(bbox)
                ax.text(
                    x + w / 2, curr_y, text,
                    fontsize=f_size, fontweight='bold',
                    ha='center', va='center', color='white', zorder=z + 2
                )
            else:
                f_size = item.get('size', 5.85)
                t_color = item.get('color', C_MUTED)
                ax.text(
                    x + w / 2, curr_y, text,
                    fontsize=f_size, ha='center', va='center',
                    color=t_color, zorder=z + 1
                )

    # Top Band Cards (Stage 1)
    draw_card(2.4, 56.5, 22.0, 36.2, [
        {'type': 'title', 'text': '1. Unseen Instance &'},
        {'type': 'title', 'text': 'Mean-Field Continuation'},
        {'type': 'subtitle', 'text': 'Input (J, h) Only | 0 Training Data', 'color': C_NAVY},
        {'type': 'badge', 'text': '93.8%–99.6% SPIN ACCURACY', 'color': C_NAVY},
        {'type': 'text', 'text': r'Damped MF: $\beta_{\mathrm{mf}} \in \{2.5, 6.0, 16.0\}$'},
        {'type': 'text', 'text': 'Single-edge cluster swap sweeps'},
        {'type': 'text', 'text': r'Self-consistent $\hat{\mathbf{s}} \in \{-1, +1\}^n$'},
    ], border=C_NAVY)

    draw_card(26.8, 56.5, 22.0, 36.2, [
        {'type': 'title', 'text': '2. 1D Daubechies'},
        {'type': 'title', 'text': 'Convex-Interval Anchors'},
        {'type': 'subtitle', 'text': r'$F H_J F = H_J,\; F H_h F = -H_h$', 'color': C_PURPLE},
        {'type': 'badge', 'text': 'STRICT INTERVAL CONVEXITY', 'color': C_PURPLE},
        {'type': 'text', 'text': r'Daubechies flat-top kernel $I_q(k, k)$'},
        {'type': 'text', 'text': 'Zero-free singularity intervals'},
        {'type': 'text', 'text': r'Bisection roots $(\mathbf{g}^{(0)}, \mathbf{\theta}^{(0)})$'},
    ], border=C_PURPLE)

    draw_card(51.2, 56.5, 22.0, 36.2, [
        {'type': 'title', 'text': '3. Staged O(kn) SU(2)'},
        {'type': 'title', 'text': 'Adjoint Refinement'},
        {'type': 'subtitle', 'text': 'Step 1a (kD) → Step 1b (2kD)', 'color': C_TEAL},
        {'type': 'badge', 'text': 'MIXER BOUND |θⱼ| ≤ 1.40 < π/2', 'color': C_TEAL},
        {'type': 'text', 'text': r'Step 1a: $k$D cost angles $\mathbf{g} \in \mathbb{R}^k$'},
        {'type': 'text', 'text': r'Step 1b: $2k$D full pair $(\mathbf{g}, \mathbf{\theta}) \in \mathbb{R}^{2k}$'},
        {'type': 'text', 'text': r'Prevents $\pi/2$ spin-flip collapse'},
    ], border=C_TEAL)

    draw_card(75.6, 56.5, 22.0, 36.2, [
        {'type': 'title', 'text': '4. Quantum-Classical'},
        {'type': 'title', 'text': 'Energy Identity'},
        {'type': 'subtitle', 'text': r'$\langle\psi_p|H_C|\psi_p\rangle \equiv E_{\mathrm{prod}}(\mathbf{g}, \mathbf{\theta})$', 'color': C_TEAL},
        {'type': 'badge', 'text': '0 QUANTUM EVALUATIONS', 'color': C_TEAL},
        {'type': 'text', 'text': 'Exact unentangled product state'},
        {'type': 'text', 'text': r'Inversion guard: $h_i^{\mathrm{eff}}(\mathbf{m}) = h_i + (J\mathbf{m})_i$'},
        {'type': 'text', 'text': r'Outputs $2p$-D angles $\vartheta_0 \in \mathbb{R}^{2p}$'},
    ], border=C_TEAL, bg='#F0FDF4')

    # Bottom Band Cards (Stage 2)
    draw_card(2.4, 4.5, 22.0, 36.2, [
        {'type': 'title', 'text': '5. Global 2p-D QAOA'},
        {'type': 'title', 'text': 'Hardware-Native Ansatz'},
        {'type': 'subtitle', 'text': r'Warm-Started at $\vartheta_0 \in \mathbb{R}^{2p}$', 'color': C_TEAL},
        {'type': 'badge', 'text': 'GLOBAL MIXER PRESERVED', 'color': C_TEAL},
        {'type': 'text', 'text': r'Uniform initial state $|+\rangle^{\otimes n}$'},
        {'type': 'text', 'text': r'Global pulse $U_B(\beta_\ell) = e^{-i\beta_\ell\sum X_i}$'},
        {'type': 'text', 'text': 'Rydberg / trapped-ion native'},
    ], border=C_TEAL)

    draw_card(26.8, 4.5, 22.0, 36.2, [
        {'type': 'title', 'text': '6. Symmetry-Breaking'},
        {'type': 'title', 'text': 'Coupling Reactivation'},
        {'type': 'subtitle', 'text': r'Perturb $(-g_j + \delta_j,\; g_j + \delta_j)$', 'color': C_AMBER},
        {'type': 'badge', 'text': 'LINEAR Hⱼ RE-ACTIVATION', 'color': C_AMBER},
        {'type': 'text', 'text': r'Reactivates $e^{-2i\delta_j H_J}$ at $O(\delta_j)$'},
        {'type': 'text', 'text': 'Breaks intra-pair anti-symmetry'},
        {'type': 'text', 'text': 'Non-vanishing tangent gradient'},
    ], border=C_AMBER)

    draw_card(51.2, 4.5, 22.0, 36.2, [
        {'type': 'title', 'text': '7. Bound-Constrained'},
        {'type': 'title', 'text': 'L-BFGS-B Recoupling'},
        {'type': 'subtitle', 'text': r'$\min_\vartheta \langle\psi_p(\vartheta)| H_C |\psi_p(\vartheta)\rangle$', 'color': C_AMBER},
        {'type': 'badge', 'text': '45.2% FEWER QUANTUM CALLS', 'color': C_AMBER},
        {'type': 'text', 'text': 'Starts deep in target quadrant'},
        {'type': 'text', 'text': '64.8 vs. 118.3 (TQA) grad evals'},
        {'type': 'text', 'text': 'Fast local multi-body convergence'},
    ], border=C_AMBER)

    draw_card(75.6, 4.5, 22.0, 36.2, [
        {'type': 'title', 'text': '8. True Multi-Body'},
        {'type': 'title', 'text': 'Ground State Convergence'},
        {'type': 'subtitle', 'text': r'$P_{\mathrm{gs}} = 46.16\%$ at $n=10$ ($2.06\times$ TQA)', 'color': C_TEAL},
        {'type': 'badge', 'text': '100% SHOT-NOISE IMMUNITY', 'color': C_TEAL},
        {'type': 'text', 'text': r'Reaches $60.80\%$ $P_{\mathrm{gs}}$ at $p = 12$'},
        {'type': 'text', 'text': r'$45.7\%$ coverage ($P_{\mathrm{gs}} \geq 50\%$)'},
        {'type': 'text', 'text': 'Verified across 5,120 instances'},
    ], border=C_TEAL, bg='#F0FDF4')

    def draw_arrow(x1, y1, x2, y2, color=C_DARK, lw=1.15, style='->', ls='-'):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style, color=color, lw=lw, linestyle=ls,
                shrinkA=1.5, shrinkB=1.5, mutation_scale=10
            ),
            zorder=6
        )

    # Top arrows
    draw_arrow(24.4, 74.6, 26.8, 74.6, color=C_NAVY, lw=1.2)
    draw_arrow(48.8, 74.6, 51.2, 74.6, color=C_PURPLE, lw=1.2)
    draw_arrow(73.2, 74.6, 75.6, 74.6, color=C_TEAL, lw=1.2)

    # Bottom arrows
    draw_arrow(24.4, 22.6, 26.8, 22.6, color=C_TEAL, lw=1.2)
    draw_arrow(48.8, 22.6, 51.2, 22.6, color=C_AMBER, lw=1.2)
    draw_arrow(73.2, 22.6, 75.6, 22.6, color=C_AMBER, lw=1.25)

    # Stage 1 -> Stage 2 connector corridor (clear of Stage 2 title at x >= 17.0)
    corridor_x = [86.6, 86.6, 11.5, 11.5]
    corridor_y = [56.5, 50.2, 50.2, 40.7]
    ax.plot(corridor_x[:-1], corridor_y[:-1], color=C_TEAL, lw=1.15, linestyle='--', zorder=6)
    ax.annotate(
        '', xy=(corridor_x[-1], corridor_y[-1]), xytext=(corridor_x[-2], corridor_y[-2]),
        arrowprops=dict(
            arrowstyle='->', color=C_TEAL, lw=1.15, linestyle='--',
            shrinkA=0, shrinkB=1.5, mutation_scale=9
        ),
        zorder=6
    )

    ax.text(
        50.0, 50.2,
        r'Zero-Measurement Starting Angles Hand-Off:  $\vartheta_0 \in \mathbb{R}^{2p}$  (Initial Product $P_{\mathrm{gs}}^{(0)} = 41.77\% > 1.86\times$ Final TQA)',
        fontsize=6.3, ha='center', va='center',
        color=C_TEAL, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=C_TEAL, lw=0.7),
        zorder=7
    )

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    pdf_path = OUT / 'fig1_echo_mf_2kd_overview.pdf'
    svg_path = OUT / 'fig1_echo_mf_2kd_overview.svg'
    png_path = OUT / 'fig1_echo_mf_2kd_overview.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    svg_content = svg_path.read_text(encoding='utf-8')
    svg_clean = re.sub(r'<!DOCTYPE[\s\S]*?>', '', svg_content)
    svg_path.write_text(svg_clean, encoding='utf-8')
    print(f"Generated Paper 2 Figure 1:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == '__main__':
    create_fig1()
