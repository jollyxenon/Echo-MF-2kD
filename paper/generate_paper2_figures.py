#!/usr/bin/env python3
"""生成 Echo-MF-2kD 数据主图与源数据 CSV（输出至 paper/figures/）。"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from learning.aggregation import interval

OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
QA = ROOT / 'runs/canonical/figure-qa'
QA.mkdir(parents=True, exist_ok=True)
CANON = ROOT / 'runs/canonical'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.dpi': 600,
    'figure.facecolor': 'white',
})


def write_csv(name, rows):
    """保存与图形相同统计单位的源数据。"""
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def label(ax, letter, title):
    """固定物理偏移放置面板编号与短标题。"""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-25, 13), textcoords='offset points',
                fontsize=10, fontweight='bold', annotation_clip=False)
    ax.set_title(title, loc='left', pad=13)


def save(fig, name):
    """输出可编辑矢量和高分辨率预览。"""
    fig.canvas.draw()
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.svg'))
    fig.savefig(OUT / (name + '.png'), dpi=600)
    plt.close(fig)


# =============================================================================
# 1. 免训练启发式赛道与先验阶梯对比图 (mf-echo-comparison, Paper 2 Fig. 2)
# =============================================================================
REV = CANON / 'baselines'


def load_rev_jsonl(rel_path, expected_ids):
    """加载基线逐题记录并校验题号对齐。"""
    rows = [json.loads(line) for line in (REV / rel_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows.sort(key=lambda r: str(r['id']))
    assert [r['id'] for r in rows] == expected_ids
    return (np.array([r['initial_probability'] for r in rows]) * 100,
            np.array([r['probability'] for r in rows]) * 100)


if (CANON / 'n10_baselines_summary.json').exists() and all(
    (CANON / f'n10_2kd_test1024_p{p:02d}_v2.jsonl').exists() for p in (8, 10, 12)
):
    expected_ids = json.loads((CANON / 'n10_baselines_summary.json').read_text(encoding='utf-8'))['test_ids']
    mf_paired_rows = []
    REV_SPECS = [
        ('Echo-MF-Flat', 'Echo-MF-Flat__p{p:02d}.jsonl'),
        ('Echo-MF-End', 'Echo-MF-End__p{p:02d}.jsonl'),
        ('Echo-MF-Alt', 'Echo-MF-Alt__p{p:02d}.jsonl'),
        ('TQA', 'TQA__p{p:02d}.jsonl'),
        ('Echo-Energy', 'Echo-Energy__p{p:02d}.jsonl'),
        ('RI (1)', 'RI-1__p{p:02d}.jsonl'),
    ]
    for tag, pattern in REV_SPECS:
        for p in (8, 10, 12):
            init_v, final_v = load_rev_jsonl(pattern.format(p=p), expected_ids)
            for inst_id, x, y in zip(expected_ids, init_v, final_v):
                mf_paired_rows.append(dict(id=inst_id, n=10, depth=p, method=tag, initial_percent=x, final_percent=y))
    for p in (8, 10, 12):
        p_path = CANON / f'n10_2kd_test1024_p{p:02d}_v2.jsonl'
        rows_2kd = [json.loads(line) for line in p_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows_2kd.sort(key=lambda r: str(r['id']))
        assert [r['id'] for r in rows_2kd] == expected_ids
        init_2kd = np.array([r['Echo-MF-2kD']['top1']['initial_probability'] for r in rows_2kd]) * 100
        final_2kd = np.array([r['Echo-MF-2kD']['top1']['probability'] for r in rows_2kd]) * 100
        for inst_id, x, y in zip(expected_ids, init_2kd, final_2kd):
            mf_paired_rows.append(dict(id=inst_id, n=10, depth=p, method='Echo-MF-2kD', initial_percent=x, final_percent=y))
    write_csv('mf-echo-paired.csv', mf_paired_rows)

mf_data = {}
with (OUT / 'mf-echo-paired.csv').open('r', encoding='utf-8') as f:
    grouped_paired = {}
    for row in csv.DictReader(f):
        key = (row['method'], int(row['depth']))
        grouped_paired.setdefault(key, []).append((row['id'], float(row['initial_percent']), float(row['final_percent'])))
    for key, items in grouped_paired.items():
        items.sort(key=lambda t: t[0])
        mf_data[key] = (np.array([t[1] for t in items]), np.array([t[2] for t in items]))

fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6))
fig.subplots_adjust(left=.145, right=.975, bottom=.08, top=.91, wspace=.36, hspace=.40)
ma, mb, mc, md = axes.flat
label(ma, 'a', 'Training-free ladder & refinement')
label(mb, 'b', 'Training-free depth scaling')
label(mc, 'c', 'All 1,024 test instances (ECDF)')
label(md, 'd', 'Paired gain of Echo-MF-2kD')
mf_stats = []

LADDER_ORDER = [
    ('Echo-MF-2kD', 'Echo-MF-2kD', '#007F78'),
    ('Echo-MF-Flat', 'Echo-MF-Flat', '#1F9E89'),
    ('TQA', 'TQA', '#4676A8'),
    ('Echo-Energy', 'Echo-Energy', '#8E969D'),
    ('RI (1)', 'RI (1)', '#B8BCC2'),
]
for i, (key, disp, col) in enumerate(LADDER_ORDER):
    before = np.mean([mf_data[key, p][0] for p in (8, 10, 12)], axis=0)
    after = np.mean([mf_data[key, p][1] for p in (8, 10, 12)], axis=0)
    m0, ci0 = interval(before)
    m1, ci1 = interval(after)
    gain = m1 - m0
    ma.plot([m0, m1], [i, i], color=col, lw=2.0, alpha=.85, zorder=2)
    ma.errorbar(m0, i, xerr=[[m0 - ci0[0]], [ci0[1] - m0]], fmt='o', mfc='white', mec=col,
                mew=1.2, color=col, ms=4.8, capsize=2, zorder=3)
    ma.errorbar(m1, i, xerr=[[m1 - ci1[0]], [ci1[1] - m1]], fmt='o', mfc=col, mec=col,
                color=col, ms=5.2, capsize=2, zorder=3)
    ma.text(ci0[0] - 1.1, i, f'{m0:.1f}%', va='center', ha='right', fontsize=6.4, color='#5A646D')
    ma.text(ci1[1] + 1.1, i, f'{m1:.1f}%', va='center', ha='left', fontsize=6.9,
            fontweight='bold' if i == 0 else 'normal', color=col)
    mid_x = (ci0[1] + ci1[0]) / 2
    ma.text(mid_x, i - .27, f'+{gain:.1f} pp', va='bottom', ha='center', fontsize=5.8,
            fontweight='bold', color=col)
    mf_stats.append(dict(method=disp, depth='mean_8_10_12', initial=m0, final=m1, ci_low=ci1[0], ci_high=ci1[1]))

ma.set(yticks=range(len(LADDER_ORDER)), yticklabels=[x[1] for x in LADDER_ORDER],
       ylim=(4.65, -.65), xlim=(-9, 64), xticks=[0, 20, 40, 60], xlabel='Ground-state probability (%)')

TF_METHODS = [
    ('Echo-MF-2kD', 'Echo-MF-2kD', '#007F78', 'o-', 1.6),
    ('Echo-MF-Flat', 'Echo-MF-Flat', '#1F9E89', 's-', 1.2),
    ('Echo-MF-End', 'Echo-MF-End', '#8774AC', '^-', 1.0),
    ('TQA', 'TQA', '#4676A8', 'v--', 1.0),
    ('Echo-Energy', 'Echo-Energy', '#8E969D', 'P:', 0.9),
]
for yv in (15, 30, 45, 60):
    mb.plot([7.5, 12.18], [yv, yv], color='#E8EBED', lw=.5, zorder=0)
for i, (key, disp, col, fmt, lw_v) in enumerate(TF_METHODS):
    means, lows, highs = [], [], []
    for p in (8, 10, 12):
        mean, ci = interval(mf_data[key, p][1])
        means.append(mean)
        lows.append(mean - ci[0])
        highs.append(ci[1] - mean)
        mf_stats.append(dict(method=disp, depth=p, initial=float(np.mean(mf_data[key, p][0])),
                             final=mean, ci_low=ci[0], ci_high=ci[1]))
    x = np.array([8, 10, 12]) + (i - 2.0) * .065
    mb.errorbar(x, means, yerr=[lows, highs], color=col, fmt=fmt, lw=lw_v, ms=3.9, capsize=1.8, label=disp)
    y_pos = means[-1]
    if key == 'Echo-MF-Flat':
        y_pos = 30.8
    elif key == 'TQA':
        y_pos = 27.3
    elif key == 'Echo-MF-End':
        y_pos = 22.8
    elif key == 'Echo-Energy':
        y_pos = 17.9
    mb.text(12.30, y_pos, f'{means[-1]:.1f}%', va='center', fontsize=6.4,
            fontweight='bold' if i == 0 else 'normal', color=col)

mb.set(xlim=(7.5, 13.3), xticks=[8, 10, 12], ylim=(6, 70), yticks=[15, 30, 45, 60],
       xlabel='Circuit depth p', ylabel='Mean probability (%)')

mf2k_cross = np.mean([mf_data['Echo-MF-2kD', p][1] for p in (8, 10, 12)], axis=0)
mf_cross = np.mean([mf_data['Echo-MF-Flat', p][1] for p in (8, 10, 12)], axis=0)
tqa1_cross = np.mean([mf_data['TQA', p][1] for p in (8, 10, 12)], axis=0)

for vals, col, disp, ls, lw_v in [
    (mf2k_cross, '#007F78', 'Echo-MF-2kD', '-', 1.8),
    (mf_cross, '#1F9E89', 'Echo-MF-Flat', '--', 1.3),
    (tqa1_cross, '#4676A8', 'TQA', '-.', 1.2),
]:
    sv = np.sort(vals)
    cdf = np.arange(1, len(sv) + 1) / len(sv)
    mc.step(sv, cdf, color=col, ls=ls, lw=lw_v, where='post', label=disp, zorder=3)

mc.axvline(50, color='#9BA3AA', lw=.7, ls='--', zorder=1)
pct_mf2k50 = np.mean(mf2k_cross >= 50) * 100
pct_mf50 = np.mean(mf_cross >= 50) * 100
pct_tqa1_50 = np.mean(tqa1_cross >= 50) * 100
mc.text(52, .50, r'$P_{\mathrm{gs}} \geq 50\%$:', va='bottom', ha='left', fontsize=7.3, color='#2D3748')
mc.text(52, .31, f'Echo-MF-2kD: {pct_mf2k50:.1f}%\nEcho-MF-Flat: {pct_mf50:.1f}%\nTQA: {pct_tqa1_50:.1f}%',
        va='bottom', ha='left', fontsize=6.1, color='#2D3748')
mc.set(xlim=(0, 100), ylim=(0, 1.02), xticks=[0, 25, 50, 75, 100], yticks=[0, .25, .5, .75, 1.0],
       xlabel='Ground-state probability (%)', ylabel='Empirical CDF')
mc.legend(loc='lower right', bbox_to_anchor=(.99, .02), frameon=False, fontsize=6.1)

for p in (8, 10, 12):
    d_tqa1 = mf_data['Echo-MF-2kD', p][1] - mf_data['TQA', p][1]
    d_mf1d = mf_data['Echo-MF-2kD', p][1] - mf_data['Echo-MF-Flat', p][1]
    for k, (diff, col, disp) in enumerate([(d_tqa1, '#4676A8', 'vs. TQA'),
                                           (d_mf1d, '#1F9E89', 'vs. Echo-MF-Flat')]):
        pos = p + (k - .5) * .64
        clipped = np.clip(diff, -42, 85)
        parts = md.violinplot(clipped, positions=[pos], widths=.46, showextrema=False, orientation='vertical')
        for pc in parts['bodies']:
            pc.set_facecolor(col)
            pc.set_edgecolor('none')
            pc.set_alpha(.30)
        q25, q50, q75 = np.percentile(diff, [25, 50, 75])
        md.plot([pos, pos], [q25, q75], color=col, lw=2.2, solid_capstyle='round')
        m_val, ci = interval(diff)
        md.errorbar(pos, m_val, yerr=[[m_val - ci[0]], [ci[1] - m_val]], fmt='o', color='white',
                    markeredgecolor=col, markeredgewidth=1.2, ms=3.8, capsize=1.5, zorder=4)
        md.text(pos, 86, f'+{m_val:.1f}', va='bottom', ha='center', fontsize=5.6,
                fontweight='bold' if k == 0 else 'normal', color=col)
        mf_stats.append(dict(method='Echo-MF-2kD ' + disp, depth=p, initial='', final=m_val, ci_low=ci[0], ci_high=ci[1]))

md.axhline(0, color='#9BA3AA', lw=.7, ls='--')
md.set(xlim=(7.1, 12.9), xticks=[8, 10, 12], ylim=(-46, 108), yticks=[-30, 0, 30, 60, 90],
       xlabel='Circuit depth p', ylabel='Paired difference (pp)')
md.legend(handles=[
    Line2D([], [], color='#4676A8', lw=2.2, label='vs. TQA (32 evals)'),
    Line2D([], [], color='#1F9E89', lw=2.2, label='vs. Echo-MF-Flat (1D)'),
], loc='upper left', bbox_to_anchor=(.01, 1.01), ncol=2, columnspacing=.6, handlelength=1.2,
    frameon=False, fontsize=5.8)

fig.legend(handles=[Line2D([], [], marker=fmt[0], color=co, ls=fmt[1:], label=lb)
                    for _, lb, co, fmt, _ in TF_METHODS],
           loc='upper center', bbox_to_anchor=(.54, .995), ncol=5, columnspacing=.8,
           handletextpad=.4, frameon=False, fontsize=6.8)
save(fig, 'mf-echo-comparison')
write_csv('mf-echo-summary.csv', mf_stats)


# =============================================================================
# 2. 跨规模扩展与散粒噪声鲁棒性主图 (scale-and-shot-noise, Paper 2 Fig. 3)
# =============================================================================
C_PRIOR = '#007F78'
C_MF = '#1F9E89'
C_TQA = '#4676A8'

if (CANON / 'mwis_multiscalar_scaling_p10.json').exists():
    mwis_scale = json.loads((CANON / 'mwis_multiscalar_scaling_p10.json').read_text(encoding='utf-8'))
    online_scale = json.loads((CANON / 'online_kd_scaling_p10.json').read_text(encoding='utf-8'))
    shot_summary = json.loads((CANON / 'shot_noise_summary_p10.json').read_text(encoding='utf-8'))
    csv_rows = []
    for n in sorted(int(k[1:]) for k in mwis_scale.keys()):
        rec = mwis_scale[f'n{n:02d}']
        for m_label, k_pct, k_ci in [
            ('Echo-Flat-2kD (Top-1)', 'Echo-Flat-2kD_top1_pct', 'Echo-Flat-2kD_top1_ci_offset'),
            ('Echo-Flat-1D (Top-1)', 'Echo-Flat-1D_top1_pct', 'Echo-Flat-1D_top1_ci_offset'),
            ('TQA (Top-1)', 'TQA_top1_pct', 'TQA_top1_ci_offset'),
        ]:
            lo_v, hi_v = map(float, rec[k_ci])
            csv_rows.append(dict(panel='a_cross_scale_mwis_p10', setting=f'n={n}', method=m_label,
                                 mean_pct=round(float(rec[k_pct]), 2), ci_minus=round(lo_v, 2), ci_plus=round(hi_v, 2)))
    for n in sorted(int(k[1:]) for k in online_scale.keys()):
        r = online_scale[f'n{n:02d}']
        csv_rows.append(dict(panel='b_oriented_spin_accuracy', setting=f'n={n}', method='Echo-MF-2kD',
                             mean_pct=round(float(r['mf_kd_bit_acc_pct']), 2), ci_minus=0.0, ci_plus=0.0))
        for m_label, k_pct, k_ci in [
            ('Echo-MF-2kD', 'mf_kd_final_pct', 'mf_kd_final_ci'),
            ('Echo-MF-Flat (1D)', 'mf_1d_final_pct', 'mf_1d_final_ci'),
            ('Online TQA (32 evals)', 'tqa_final_pct', 'tqa_final_ci'),
        ]:
            csv_rows.append(dict(panel='b_online_scaling_p10', setting=f'n={n}', method=m_label,
                                 mean_pct=round(float(r[k_pct]), 2),
                                 ci_minus=round(float(r[k_ci][0]), 2),
                                 ci_plus=round(float(r[k_ci][1]), 2)))
    for s in [16, 32, 64, 128, 256, 512, 1024]:
        m1 = float(shot_summary['regime_a_stage1_shot_noise']['TQA'][str(s)]['exact_candidate_match_pct'])
        csv_rows.append(dict(panel='c_stage1_shot_noise_hit_rate', setting=f'N_shots={s}', method='TQA',
                             mean_pct=round(m1, 2), ci_minus=0.0, ci_plus=0.0))
    regime_b = shot_summary['regime_b_end_to_end_parameter_shift']
    for m_name, raw_m in [
        ('Echo-MF-2kD (64 steps)', 'Echo-MF-2kD'),
        ('Echo-MF-Flat (64 steps)', 'Echo-MF-Flat'),
        ('TQA (64 steps)', 'TQA'),
    ]:
        for s_val in [64, 256, 1024, 4096]:
            m_val = float(regime_b[str(s_val)][raw_m]['mean_final_pct'])
            l_val, h_val = map(float, regime_b[str(s_val)][raw_m]['final_ci'])
            csv_rows.append(dict(panel='d_end_to_end_finite_shot', setting=f'N_shots={s_val}', method=m_name,
                                 mean_pct=round(m_val, 2), ci_minus=round(l_val, 2), ci_plus=round(h_val, 2)))
    write_csv('scale-and-shot-noise-summary.csv', csv_rows)

with (OUT / 'scale-and-shot-noise-summary.csv').open('r', encoding='utf-8') as f:
    scale_rows = list(csv.DictReader(f))

scale_lookup = {(r['panel'], r['setting'], r['method']): (float(r['mean_pct']), float(r['ci_minus']), float(r['ci_plus']))
                for r in scale_rows}
sizes_all = sorted({int(r['setting'].split('=')[1]) for r in scale_rows if r['panel'] == 'a_cross_scale_mwis_p10'})
echo_kd_means = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-2kD (Top-1)'][0] for n in sizes_all]
echo_kd_lo = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-2kD (Top-1)'][1] for n in sizes_all]
echo_kd_hi = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-2kD (Top-1)'][2] for n in sizes_all]
echo_1d_means = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-1D (Top-1)'][0] for n in sizes_all]
echo_1d_lo = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-1D (Top-1)'][1] for n in sizes_all]
echo_1d_hi = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'Echo-Flat-1D (Top-1)'][2] for n in sizes_all]
tqa_means = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'TQA (Top-1)'][0] for n in sizes_all]
tqa_lo = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'TQA (Top-1)'][1] for n in sizes_all]
tqa_hi = [scale_lookup['a_cross_scale_mwis_p10', f'n={n}', 'TQA (Top-1)'][2] for n in sizes_all]

online_sizes = sorted({int(r['setting'].split('=')[1]) for r in scale_rows if r['panel'] == 'b_online_scaling_p10'})
mf_accs = [scale_lookup['b_oriented_spin_accuracy', f'n={n}', 'Echo-MF-2kD'][0] for n in online_sizes]
mf_p10_bars = [scale_lookup['b_online_scaling_p10', f'n={n}', 'Echo-MF-2kD'][0] for n in online_sizes]
mf1d_p10_bars = [scale_lookup['b_online_scaling_p10', f'n={n}', 'Echo-MF-Flat (1D)'][0] for n in online_sizes]
tqa_p10_bars = [scale_lookup['b_online_scaling_p10', f'n={n}', 'Online TQA (32 evals)'][0] for n in online_sizes]

fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6))
fig.subplots_adjust(left=.115, right=.975, bottom=.10, top=.91, wspace=.34, hspace=.42)
ax_a, ax_b, ax_c, ax_d = axes.flat

label(ax_a, 'a', f'Cross-scale MWIS scaling (n = {sizes_all[0]}..{sizes_all[-1]})')
label(ax_b, 'b', 'Oriented spin accuracy & online scaling')
label(ax_c, 'c', 'Stage-1 selection under projective shot noise')
label(ax_d, 'd', 'End-to-end finite-shot optimization')

x_a = np.arange(len(sizes_all))
ax_a.errorbar(x_a, echo_kd_means, yerr=[echo_kd_lo, echo_kd_hi], fmt='o-', color=C_PRIOR, lw=1.5, ms=4.4,
              capsize=2.1, capthick=0.8, label='Echo-Flat-2kD (Top-1)')
ax_a.errorbar(x_a, echo_1d_means, yerr=[echo_1d_lo, echo_1d_hi], fmt='s:', color=C_MF, lw=1.25, ms=3.8,
              capsize=1.9, capthick=0.7, label='Echo-Flat-1D (Top-1)')
ax_a.errorbar(x_a, tqa_means, yerr=[tqa_lo, tqa_hi], fmt='v--', color=C_TQA, lw=1.3, ms=4.1,
              capsize=2.0, capthick=0.75, label='TQA (Top-1)')
ax_a.fill_between(x_a, tqa_means, echo_kd_means, color=C_PRIOR, alpha=0.10)
for idx_show in range(len(sizes_all)):
    dx_a = 0.14 if idx_show == 0 else (0.10 if idx_show < len(sizes_all) - 1 else -0.04)
    ax_a.text(x_a[idx_show] + dx_a, echo_kd_means[idx_show] + echo_kd_hi[idx_show] + 2.2,
              f'{echo_kd_means[idx_show]:.1f}%', ha='center', va='bottom', fontsize=6.3, color=C_PRIOR, fontweight='bold')
ax_a.set_xlim(-0.36, len(sizes_all) - 0.62)
ax_a.set_xticks(x_a)
ax_a.set_xticklabels([str(n) for n in sizes_all])
ax_a.set_xlabel(f'Number of qubits n ({1 << sizes_all[0]} to {1 << sizes_all[-1]:,} states)')
ax_a.set_ylabel('Ground-state probability (%)')
ax_a.set_ylim(0, 112)
ax_a.legend(loc='upper right', frameon=False, fontsize=6.1)

xb = np.arange(len(online_sizes))
ax_b.plot(xb, mf_accs, 'o-', color=C_PRIOR, lw=1.5, ms=4.0, label='Echo-MF-2kD spin acc. (%)')

w_bar = 0.25
ax_b.bar(xb - w_bar, mf_p10_bars, width=w_bar, color=C_PRIOR, alpha=0.88, label='Echo-MF-2kD prob. (0 evals)')
ax_b.bar(xb, mf1d_p10_bars, width=w_bar, color=C_MF, alpha=0.85, label='Echo-MF-Flat prob. (0 evals)')
ax_b.bar(xb + w_bar, tqa_p10_bars, width=w_bar, color=C_TQA, alpha=0.85, label='Online TQA prob. (32 evals)')

ax_b.set_xticks(xb)
ax_b.set_xticklabels([str(n) for n in online_sizes])
ax_b.set_xlabel('Number of qubits n (online unsupervised)')
ax_b.set_ylabel('Spin accuracy / Online prob. (%)')
ax_b.set_ylim(0, 109)
ax_b.legend(loc='center right', bbox_to_anchor=(1.01, 0.62), frameon=False, fontsize=5.8, handlelength=1.4, labelspacing=0.30)

shots_c = [16, 32, 64, 128, 256, 512, 1024]
xc = np.arange(len(shots_c))
tqa_match = [scale_lookup['c_stage1_shot_noise_hit_rate', f'N_shots={s}', 'TQA'][0] for s in shots_c]

ax_c.axhline(100.0, color=C_PRIOR, lw=1.6, ls='-', label='Echo-MF (0 shots, 100%)')
ax_c.plot(xc, tqa_match, 'v-', color=C_TQA, lw=1.5, ms=4.5, label='TQA (32 candidates)')

ax_c.text(xc[0] + 0.15, tqa_match[0] - 4.5, f'{tqa_match[0]:.1f}%', ha='center', va='top', fontsize=6.4, color=C_TQA)
ax_c.text(xc[2] - 0.15, tqa_match[2] + 3.5, f'{tqa_match[2]:.1f}%', ha='center', va='bottom', fontsize=6.4, color=C_TQA)
ax_c.text(xc[4], tqa_match[4] + 3.5, f'{tqa_match[4]:.1f}%', ha='center', va='bottom', fontsize=6.4, color=C_TQA)
ax_c.text(xc[6], tqa_match[6] + 3.0, f'{tqa_match[6]:.1f}%', ha='center', va='bottom', fontsize=6.4, color=C_TQA)
ax_c.set_xticks(xc)
ax_c.set_xticklabels([str(s) for s in shots_c])
ax_c.set_xlabel('Stage-1 projective measurements per circuit')
ax_c.set_ylabel('Optimal Stage-1 candidate hit rate (%)')
ax_c.set_ylim(20, 112)
ax_c.legend(loc='lower right', bbox_to_anchor=(1.0, 0.04), frameon=False, fontsize=6.5)

shots_d = [64, 256, 1024, 4096]
xd = np.arange(len(shots_d))
plotted_series = {}
for m_name, col, fmt, disp in [
    ('Echo-MF-2kD (64 steps)', C_PRIOR, 'o-', 'Echo-MF-2kD (64st, 0+2625)'),
    ('Echo-MF-Flat (64 steps)', C_MF, 's:', 'Echo-MF-Flat (64st, 0+2625)'),
    ('TQA (64 steps)', C_TQA, 'v--', 'TQA (64st, 32+2625)'),
]:
    means = [scale_lookup['d_end_to_end_finite_shot', f'N_shots={s}', m_name][0] for s in shots_d]
    lo = [scale_lookup['d_end_to_end_finite_shot', f'N_shots={s}', m_name][1] for s in shots_d]
    hi = [scale_lookup['d_end_to_end_finite_shot', f'N_shots={s}', m_name][2] for s in shots_d]
    plotted_series[m_name] = (means, lo, hi)
    ax_d.errorbar(xd, means, yerr=[lo, hi], fmt=fmt, color=col, lw=1.35, ms=4.1,
                  capsize=2.0, capthick=0.75, label=disp)

mf2k_means, _, mf2k_hi = plotted_series['Echo-MF-2kD (64 steps)']
mff_means_d, mff_lo_d, _ = plotted_series['Echo-MF-Flat (64 steps)']
tqa_means_d, _, tqa_hi_d = plotted_series['TQA (64 steps)']
ax_d.text(xd[-1], mf2k_means[-1] + mf2k_hi[-1] + 0.9, f'{mf2k_means[-1]:.2f}%',
          ha='center', va='bottom', fontsize=6.4, color=C_PRIOR, fontweight='bold')
ax_d.text(xd[0] + 0.16, mf2k_means[0] + mf2k_hi[0] + 0.9, f'{mf2k_means[0]:.2f}%',
          ha='center', va='bottom', fontsize=6.4, color=C_PRIOR, fontweight='bold')
ax_d.text(xd[-1], tqa_means_d[-1] + tqa_hi_d[-1] + 0.9, f'{tqa_means_d[-1]:.2f}%',
          ha='center', va='bottom', fontsize=6.2, color=C_TQA)
ax_d.text(xd[-1], mff_means_d[-1] - mff_lo_d[-1] - 0.9, f'{mff_means_d[-1]:.2f}%',
          ha='center', va='top', fontsize=6.2, color=C_MF)
ax_d.set_xticks(xd)
ax_d.set_xticklabels([str(s) for s in shots_d])
ax_d.set_xlabel('Projective measurements per evaluation')
ax_d.set_ylabel('Final ground-state probability (%)')
ax_d.set_ylim(10, 56)
ax_d.legend(loc='center right', bbox_to_anchor=(1.0, 0.56), frameon=False,
            fontsize=5.9, handlelength=1.4)

save(fig, 'scale-and-shot-noise')
print("Successfully generated Paper 2 figures and CSVs in", OUT)
