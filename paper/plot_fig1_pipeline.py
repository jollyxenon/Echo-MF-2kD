#!/usr/bin/env python3
"""
生成 Paper 2 图 1：2k 维自旋回波奇偶消耦、精确量子—经典直积能量恒等式、
免训练平均场伴随流形精修（Echo-MF-2kD）与全角度能量重耦合总览图。
遵循 Nature Computational Science 出版规范（宽 175 mm，全部渲染字符 >= 5.0 pt，
Computer Modern 标准数学字体，双色语义体系，纯矢量零碰撞）。
"""

from pathlib import Path
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5,
    "mathtext.fontset": "cm",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
})


def daubechies_convex_nll(e: np.ndarray, k: int = 4) -> np.ndarray:
    """计算 k=4 平顶 Echo 在无零点开区间 (-pi, pi) 上的严格凸单比特负对数似然。"""
    q = np.clip(0.5 * (1.0 + np.cos(e)), 1e-6, 1.0)
    y = 1.0 - q
    p_k = (q ** k) * (1.0 + 4.0 * y + 10.0 * (y ** 2) + 20.0 * (y ** 3))
    return -np.log(np.clip(p_k, 1e-8, 1.0))


def create_fig1() -> None:
    """绘制并导出 Paper 2 图 1 架构与物理机制总览图。"""
    mm_to_inch = 1.0 / 25.4
    fig_w, fig_h = 175.0 * mm_to_inch, 88.0 * mm_to_inch
    aspect = fig_w / fig_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    # 统一克制的双色 + 中性灰语义体系（与图 2、图 3 色板同源）
    C_TEAL = "#007F78"       # 核心主强调色：2kD 自旋回波乘积流形与阶段一伴随初始化
    BG_TEAL = "#F2FBF9"
    BG_TEAL_SOFT = "#F7FCFB"
    C_SLATE = "#3B6998"      # 第二阶段标识色：2p 全角度量子能量重耦合
    BG_SLATE = "#F4F8FC"
    C_INK = "#0F172A"        # 统一深墨色主文字
    C_SUB = "#334155"        # 次级深灰文字
    C_BORDER = "#94A3B8"     # 中性边框

    def add_box(x, y, w, h, fc="white", ec=C_BORDER, lw=0.85, ls="-", rounding=1.3, z=2):
        """添加统一圆角矩形框。"""
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def add_circle(cx, cy, r_x, fc=C_TEAL, ec="none", lw=0.0, z=6):
        """按画布物理宽高比补偿绘制严格正圆。"""
        ell = Ellipse(
            (cx, cy), width=2.0 * r_x, height=2.0 * r_x * aspect,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        )
        ax.add_patch(ell)
        return ell

    def add_arrow(x1, y1, x2, y2, color=C_INK, lw=1.3, ls="-", scale=9.5, z=6):
        """添加直线有向箭头（零折弯）。"""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                shrinkA=0.0, shrinkB=0.0, mutation_scale=scale,
            ),
            zorder=z,
        )

    def add_step_badge(cx, cy, num_str, color=C_TEAL, z=7):
        """在主干模块左上角绘制正圆步骤编号。"""
        add_circle(cx, cy, r_x=1.25, fc=color, z=z)
        ax.text(cx, cy, num_str, fontsize=7.2, fontweight="bold", color="white",
                ha="center", va="center", zorder=z + 1)

    # =========================================================================
    # 顶层左框 a：2kD Spin-Echo Parity Decoupling (正对下方模块 2)
    # =========================================================================
    add_box(1.0, 64.5, 47.0, 33.8, fc=BG_TEAL_SOFT, ec=C_TEAL, lw=0.85, rounding=1.4, z=1)
    ax.text(2.6, 94.6, "a. 2kD Spin-Echo Parity Decoupling",
            fontsize=7.6, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=3)
    ax.text(46.5, 94.6, r"Free $(g_j, \theta_j) \in \mathbb{R}^2$",
            fontsize=7.3, color=C_SUB, ha="right", va="center", zorder=3)

    # 绘制第 j 对完整 Echo 量子门线路示意
    y_q1, y_q2 = 85.8, 79.0
    ax.text(5.1, y_q1, r"$|+\rangle$", fontsize=7.6, color=C_INK, ha="right", va="center", zorder=3)
    ax.text(5.1, y_q2, r"$|+\rangle$", fontsize=7.6, color=C_INK, ha="right", va="center", zorder=3)
    ax.plot([5.6, 46.4], [y_q1, y_q1], color=C_SUB, lw=0.8, zorder=2)
    ax.plot([5.6, 46.4], [y_q2, y_q2], color=C_SUB, lw=0.8, zorder=2)

    gates = [
        (6.8, 8.4, r"$U_C(-g_j)$", "white", C_BORDER, C_INK),
        (16.5, 10.6, r"$U_B(\pi/2) \propto F$", "#E6F4F3", C_TEAL, C_TEAL),
        (28.4, 8.0, r"$U_C(g_j)$", "white", C_BORDER, C_INK),
        (37.7, 7.8, r"$U_B(\theta_j)$", "white", C_TEAL, C_INK),
    ]
    for gx, gw, gtxt, gfc, gec, gtc in gates:
        add_box(gx, 76.4, gw, 12.0, fc=gfc, ec=gec, lw=0.8, rounding=0.8, z=4)
        ax.text(gx + gw / 2.0, 82.4, gtxt, fontsize=7.5, color=gtc, ha="center", va="center", zorder=5)

    ax.text(
        24.5, 69.6,
        r"$F H_J F = H_J,\; F H_h F = -H_h \;\Rightarrow\; E(g_j, \theta_j) = U_B(\theta_j)\exp(-2ig_j H_h)F$",
        fontsize=7.3, color=C_INK, ha="center", va="center", zorder=3,
    )

    # =========================================================================
    # 顶层右框 b：Exact Product-Energy Identity (正对下方模块 3)
    # =========================================================================
    add_box(50.0, 64.5, 49.0, 33.8, fc=BG_TEAL_SOFT, ec=C_TEAL, lw=0.85, rounding=1.4, z=1)
    ax.text(51.6, 94.6, "b. Exact Product-Energy Identity",
            fontsize=7.6, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=3)
    ax.text(97.5, 94.6, r"$O(kn)$ $2 \times 2$ recursion",
            fontsize=7.2, color=C_SUB, ha="right", va="center", zorder=3)

    # 左子卡片：严格单比特张量直积态
    add_box(51.4, 67.5, 20.6, 22.8, fc="white", ec=C_BORDER, lw=0.75, rounding=1.0, z=2)
    ax.text(61.7, 85.8, "Product State", fontsize=7.1, fontweight="bold", color=C_INK, ha="center", va="center", zorder=3)
    ax.text(61.7, 78.8, r"$|\psi_p(g, \theta)\rangle = {\bigotimes}_{i=1}^{n} u_i^{(k)}$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=3)
    ax.text(61.7, 71.8, r"$\langle Z_i Z_j \rangle = m_i m_j\;(i \neq j)$",
            fontsize=7.3, color=C_SUB, ha="center", va="center", zorder=3)

    add_arrow(72.3, 78.8, 74.3, 78.8, color=C_TEAL, lw=1.15, scale=8.2)

    # 右子卡片：量子期望能量 === 经典直积态能量
    add_box(74.6, 67.5, 23.0, 22.8, fc=BG_TEAL_SOFT, ec=C_TEAL, lw=0.9, rounding=1.0, z=2)
    ax.text(86.1, 85.8, "Quantum–Classical Identity",
            fontsize=7.0, fontweight="bold", color=C_TEAL, ha="center", va="center", zorder=3)
    ax.text(86.1, 78.8, r"$\langle\psi_p|H_C|\psi_p\rangle \equiv E_{\mathrm{prod}}(g, \theta)$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=3)
    ax.text(86.1, 71.8, r"${\sum}_{i<j} J_{ij} m_i m_j + {\sum}_{i} h_i m_i + C$",
            fontsize=7.3, color=C_SUB, ha="center", va="center", zorder=3)

    # =========================================================================
    # 中间连接走廊（y = 48.0 .. 64.5）：零遮挡垂直箭头 + 旁置标注
    # =========================================================================
    ax.text(1.0, 56.6, "c. Training-Free Two-Stage Pipeline",
            fontsize=7.6, fontweight="bold", color=C_TEAL, ha="left", va="center", zorder=3)
    ax.text(1.0, 51.6, "Global QAOA ansatz | 0 training data",
            fontsize=6.8, color=C_SUB, ha="left", va="center", zorder=3)

    # a -> 模块 2 (x = 33.8，实线：Daubechies 最大平坦核与区间严格凸性)
    add_arrow(33.8, 64.2, 33.8, 48.4, color=C_TEAL, lw=1.25, ls="-", scale=9.0)
    ax.text(35.1, 56.6, "Daubechies flat-top",
            fontsize=6.9, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)
    ax.text(35.1, 51.6, r"$p_k(q) = I_q(k, k)$",
            fontsize=7.5, color=C_INK, ha="left", va="center", zorder=4)

    # b -> 模块 3 (x = 62.6，实线：精确经典直积能量与防反转护栏注入伴随精修)
    add_arrow(62.6, 64.2, 62.6, 48.4, color=C_TEAL, lw=1.25, ls="-", scale=9.0)
    ax.text(64.0, 56.6, "Exact classical surrogate",
            fontsize=6.9, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)
    ax.text(64.0, 51.6, r"$E_{\mathrm{prod}}(g, \theta)$ & guard $h_i^{\mathrm{eff}}(m)$",
            fontsize=7.4, color=C_INK, ha="left", va="center", zorder=4)

    # =========================================================================
    # 底层主干：STAGE 1 -> STAGE 2 SPINE (单调从左到右 1 -> 2 -> 3 -> 4)
    # =========================================================================
    box_y, box_h = 2.2, 45.8
    mid_y = box_y + 0.5 * box_h

    # --- Box 1: MF Cluster Orientation ---
    b1_x, b1_w = 1.0, 19.0
    add_box(b1_x, box_y, b1_w, box_h, fc="white", ec=C_BORDER, lw=0.9, rounding=1.4, z=2)
    add_step_badge(b1_x + 2.2, box_y + box_h - 3.6, "1", color=C_SUB)
    ax.text(b1_x + 4.1, box_y + box_h - 3.6, "MF Orientation",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    nodes = [
        (5.5, 34.8, C_TEAL), (10.5, 37.8, C_SLATE), (15.3, 34.2, C_TEAL),
        (6.2, 25.6, C_SLATE), (11.2, 23.2, C_TEAL), (15.1, 26.8, C_SLATE),
    ]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (2, 5), (1, 4)]
    for u, v in edges:
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]],
                color="#94A3B8", lw=0.95, zorder=3)
    for nx, ny, nc in nodes:
        add_circle(nx, ny, r_x=0.88, fc=nc, ec="white", lw=0.6, z=4)

    ax.text(b1_x + b1_w / 2.0, 16.8, r"Input $(J, h)$ only",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b1_x + b1_w / 2.0, 11.3, r"$\beta$-continuation + swaps",
            fontsize=7.1, color=C_SUB, ha="center", va="center", zorder=4)
    ax.text(b1_x + b1_w / 2.0, 5.8, r"Target $\hat{s} \in \{-1, +1\}^n$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)

    # 主干粗箭头 1 -> 2
    add_arrow(20.2, mid_y, 22.6, mid_y, color=C_TEAL, lw=1.7, scale=10.5)

    # --- Box 2: 1D Convex Anchor ---
    b2_x, b2_w = 22.8, 22.5
    add_box(b2_x, box_y, b2_w, box_h, fc=BG_TEAL, ec=C_TEAL, lw=1.15, rounding=1.4, z=2)
    add_step_badge(b2_x + 2.2, box_y + box_h - 3.6, "2", color=C_TEAL)
    ax.text(b2_x + 4.1, box_y + box_h - 3.6, "1D Convex Anchor",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(b2_x + b2_w / 2.0, 38.6,
            r"$\mathcal{L}_k(a) = -{\sum}_{i=1}^{n} \ln p_k(q_i(a))$",
            fontsize=7.4, color=C_INK, ha="center", va="center", zorder=4)

    # 嵌入式科学小图：无零点开区间上的严格凸负对数似然与二分求根锚点
    in_ax = ax.inset_axes([0.258, 0.188, 0.168, 0.145], zorder=4)
    a_grid = np.linspace(-2.25, 2.25, 260)
    nll_curve = 0.55 * daubechies_convex_nll(a_grid - 0.35) + 0.45 * daubechies_convex_nll(0.8 * a_grid + 0.25)
    min_idx = int(np.argmin(nll_curve))
    a_star, nll_star = a_grid[min_idx], nll_curve[min_idx]

    in_ax.axvline(-2.35, color="#94A3B8", lw=0.75, ls="--", zorder=2)
    in_ax.axvline(2.35, color="#94A3B8", lw=0.75, ls="--", zorder=2)
    in_ax.plot([a_star, a_star], [-0.15, nll_star], color=C_TEAL, lw=0.75, ls=":", zorder=2)
    in_ax.plot(a_grid, nll_curve, color=C_TEAL, lw=1.45, zorder=3)
    in_ax.plot([a_star], [nll_star], marker="o", ms=3.8, color=C_TEAL, zorder=4)
    in_ax.set_xlim(-2.6, 2.6)
    in_ax.set_ylim(-0.15, 3.6)
    in_ax.set_xticks([-2.35, a_star, 2.35])
    in_ax.set_xticklabels(["sing.", r"$a^{(0)}$", "sing."], fontsize=7.4, color=C_SUB)
    in_ax.get_xticklabels()[1].set_color(C_TEAL)
    in_ax.set_yticks([])
    in_ax.tick_params(axis="x", length=1.8, width=0.55, pad=1.5)
    in_ax.spines["top"].set_visible(False)
    in_ax.spines["right"].set_visible(False)
    in_ax.spines["left"].set_linewidth(0.6)
    in_ax.spines["bottom"].set_linewidth(0.6)
    in_ax.spines["left"].set_color("#64748B")
    in_ax.spines["bottom"].set_color("#64748B")

    ax.text(b2_x + b2_w / 2.0, 11.2, "Zero-free convex bisection",
            fontsize=6.9, color=C_SUB, ha="center", va="center", zorder=4)
    ax.text(b2_x + b2_w / 2.0, 5.8,
            r"Anchor $(g^{(0)}, \theta^{(0)})$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)

    # 主干粗箭头 2 -> 3
    add_arrow(45.5, mid_y, 47.8, mid_y, color=C_TEAL, lw=1.7, scale=10.5)

    # --- Box 3: Staged 2kD Adjoint Refinement (Hero Module) ---
    b3_x, b3_w = 48.0, 29.2
    add_box(b3_x, box_y, b3_w, box_h, fc=BG_TEAL, ec=C_TEAL, lw=1.25, rounding=1.4, z=2)
    add_step_badge(b3_x + 2.2, box_y + box_h - 3.6, "3", color=C_TEAL)
    ax.text(b3_x + 4.1, box_y + box_h - 3.6, "Staged 2kD Adjoint",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(b3_x + b3_w / 2.0, 38.6,
            r"Step 1a: $k\mathrm{D}\;g \;\rightarrow\;$ Step 1b: $2k\mathrm{D}\;(g, \theta)$",
            fontsize=7.4, color=C_INK, ha="center", va="center", zorder=4)

    add_box(b3_x + 1.3, 17.0, b3_w - 2.6, 17.8, fc="white", ec="#99CFCB", lw=0.75, rounding=0.9, z=3)
    ax.text(b3_x + b3_w / 2.0, 31.0,
            r"Mixer bound: $|\theta_j| \leq \theta_{\max} < \pi/2$",
            fontsize=7.4, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b3_x + b3_w / 2.0, 25.6,
            r"Guard: $h_i^{\mathrm{eff}}(m) = h_i + (Jm)_i$",
            fontsize=7.4, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b3_x + b3_w / 2.0, 20.2,
            r"Minimizes NLL + guard + $E_{\mathrm{prod}}$",
            fontsize=7.3, fontweight="bold", color=C_TEAL, ha="center", va="center", zorder=4)

    ax.text(b3_x + b3_w / 2.0, 11.2,
            r"0 quantum evals ($\mathrm{SU}(2)$ adjoint)",
            fontsize=7.1, fontweight="bold", color=C_TEAL, ha="center", va="center", zorder=4)
    ax.text(b3_x + b3_w / 2.0, 5.8,
            r"Start $\vartheta_0 \in \mathbb{R}^{2p}$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)

    # 主干粗箭头 3 -> 4
    add_arrow(77.4, mid_y, 79.8, mid_y, color=C_SLATE, lw=1.7, scale=10.5)

    # --- Box 4: Full-Angle Recoupling ---
    b4_x, b4_w = 80.0, 19.0
    add_box(b4_x, box_y, b4_w, box_h, fc=BG_SLATE, ec=C_SLATE, lw=1.05, rounding=1.4, z=2)
    add_step_badge(b4_x + 2.2, box_y + box_h - 3.6, "4", color=C_SLATE)
    ax.text(b4_x + 4.1, box_y + box_h - 3.6, "2p Recoupling",
            fontsize=7.1, fontweight="bold", color=C_INK, ha="left", va="center", zorder=4)

    ax.text(b4_x + b4_w / 2.0, 38.6,
            r"$\min_{\vartheta}\; \langle \psi_p(\vartheta) | H_C | \psi_p(\vartheta) \rangle$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)

    add_box(b4_x + 1.1, 21.2, b4_w - 2.2, 12.6, fc="white", ec="#93B4D4", lw=0.75, rounding=0.9, z=3)
    ax.text(b4_x + b4_w / 2.0, 30.0,
            r"$(-g_j + \delta_j,\, g_j + \delta_j)$",
            fontsize=7.5, color=C_INK, ha="center", va="center", zorder=4)
    ax.text(b4_x + b4_w / 2.0, 24.4,
            r"Restores $\exp(-2i\delta_j H_J)$",
            fontsize=7.2, color=C_SLATE, fontweight="bold", ha="center", va="center", zorder=4)

    ax.text(b4_x + b4_w / 2.0, 14.2,
            r"Releases all $2p$ angles",
            fontsize=7.0, color=C_SUB, ha="center", va="center", zorder=4)
    ax.text(b4_x + b4_w / 2.0, 6.8,
            r"Output $|\psi_p(\vartheta^*)\rangle$",
            fontsize=7.6, fontweight="bold", color=C_INK, ha="center", va="center", zorder=4)

    pdf_path = OUT / "fig1_echo_mf_2kd_overview.pdf"
    svg_path = OUT / "fig1_echo_mf_2kd_overview.svg"
    png_path = OUT / "fig1_echo_mf_2kd_overview.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    svg_content = svg_path.read_text(encoding="utf-8")
    svg_clean = re.sub(r"<!DOCTYPE[\s\S]*?>", "", svg_content)
    svg_path.write_text(svg_clean, encoding="utf-8")
    print(f"Generated Paper 2 Figure 1:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == "__main__":
    create_fig1()
