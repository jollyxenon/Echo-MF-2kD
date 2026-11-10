"""完整对 Echo 直积流形上的确定性先验定向与分阶段多标量伴随精修（Staged-2kD-Flat）。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from echo.flat_scan import flat_scan
from echo.variants import load_formula
from echo.core import G as GAMMA_MAX, angles as expand_echo_angles, spinors, start_batches
from learning.echo_decoder import decode

VARIANTS = (
    "Echo-Flat-1D",
    "Echo-Flat-kD (Cost-Only)",
    "Echo-Flat-kD (Guarded)",
    "Echo-Flat-2kD (Unconstrained)",
    "Echo-Flat-2kD",
)


def resolve_prior_direction(J: np.ndarray, h: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """从输入单比特边缘概率 pi 确定性提取目标自旋方向 s_target in {-1, +1}^n（0 次量子评估）。

    - 离线模式（pi in {0, 1}^n）：恒等还原 s_target = 2*pi - 1；
    - 在线模式（pi in (0, 1)^n）：沿最不确定自旋微扰种子执行确定性升温平均场延续与单边簇翻转零温稳定化。
    """
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    m_init = 2.0 * np.asarray(pi, dtype=np.float64) - 1.0
    if np.min(np.abs(m_init)) > 0.999:
        return np.where(m_init >= 0.0, 1.0, -1.0)
    n = len(h_arr)
    order = np.argsort(np.abs(m_init))

    # 单比特与单边耦合簇零温局部能量稳定化（至多 R_stab = 2n 轮）
    def stabilize_1edge(s_start: np.ndarray) -> tuple[np.ndarray, float]:
        curr = s_start.copy()
        for _ in range(2 * n):
            improved = False
            for i in order:
                if curr[i] * (h_arr[i] + float(J_arr[i] @ curr)) > 1e-12:
                    curr[i] = -curr[i]
                    improved = True
            if improved:
                continue
            e_curr = 0.5 * float(curr @ J_arr @ curr) + float(h_arr @ curr)
            best_de, best_cand = -1e-12, None
            for u in range(n):
                for v in range(u + 1, n):
                    if J_arr[u, v] != 0 and curr[u] != curr[v]:
                        cand = curr.copy()
                        cand[u], cand[v] = -cand[u], -cand[v]
                        for _ in range(2):
                            for w in order:
                                if cand[w] * (h_arr[w] + float(J_arr[w] @ cand)) > 1e-12:
                                    cand[w] = -cand[w]
                        de = (0.5 * float(cand @ J_arr @ cand) + float(h_arr @ cand)) - e_curr
                        if de < best_de:
                            best_de, best_cand = de, cand
            if best_cand is not None:
                curr = best_cand
                improved = True
            if not improved:
                break
        return curr, 0.5 * float(curr @ J_arr @ curr) + float(h_arr @ curr)

    # 构造初始种子与最不确定比特翻转微扰种子
    seeds = [m_init.copy()]
    for u in order[:min(6, n)]:
        perturbed = m_init.copy()
        perturbed[u] = -1.0 if perturbed[u] >= 0.0 else 1.0
        seeds.append(perturbed)

    best_key = (np.inf, np.inf)
    best_s = np.where(m_init >= 0.0, 1.0, -1.0)
    for m0 in seeds:
        m = m0.copy()
        for beta in (2.5, 6.0, 16.0):
            for _ in range(28):
                m = 0.5 * m + 0.5 * np.tanh(-beta * (h_arr + J_arr @ m))
        s_cand, e_cand = stabilize_1edge(np.where(m >= 0.0, 1.0, -1.0))
        dist = float(np.mean((s_cand - m_init) ** 2))
        key = (round(e_cand, 7), dist)
        if key < best_key:
            best_key, best_s = key, s_cand
    return best_s


def adjoint_echo_loss_and_grad(
    g: np.ndarray,
    th: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    s_target: np.ndarray,
    g0: np.ndarray,
    mix0: np.ndarray,
    lam_inv: float = 1.5,
    alpha_e: float = 1.5,
    mu_g: float = 0.005,
    mu_th: float = 0.03,
    weighted: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """完整对 Echo 直积流形上的精确 O(kn) SU(2) 伴随反向传播损失与梯度。

    严格遵循 docs/theory-and-derivations.md §6.4 与 docs/module-settings.md §6.2 复合伴随损失公式：
      1. 局部有效场翻转敏感度权重 w_i = 1 + 1.5 * |h_i + (J s_target)_i| / mean(|h + J s_target|)；
      2. w_i 加权的逐比特负对数似然与半空间象限裕度屏障；
      3. 局部有效场防反转护栏 lam_inv 与显式经典直积能量项 alpha_e * E_prod；
      4. 围绕一维平顶解析锚点 (g0, mix0) 的近端信任域正则。
    """
    g_arr = np.asarray(g, dtype=np.float64)
    th_arr = np.asarray(th, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    s_arr = np.asarray(s_target, dtype=np.float64)
    g0_arr = np.asarray(g0, dtype=np.float64)
    mix0_arr = np.asarray(mix0, dtype=np.float64)

    k, n = len(g_arr), len(h_arr)
    nz_mask = (np.abs(h_arr) > 1e-8).astype(np.float64)

    # 翻转敏感度权重 w_i
    if weighted:
        h_tar = np.abs(h_arr + J_arr @ s_arr)
        mean_h_tar = max(float(np.mean(h_tar)), 1e-8)
        w = nz_mask * (1.0 + 1.5 * (h_tar / mean_h_tar))
    else:
        w = nz_mask

    # 前向 O(kn) SU(2) 旋量递推
    u = np.ones(n, dtype=complex) / np.sqrt(2.0)
    v = u.copy()
    phis, u_tilde, v_tilde, u_next, v_next = [], [], [], [], []
    for j in range(k):
        phi = np.exp(2j * g_arr[j] * h_arr)
        ut, vt = phi * v, np.conj(phi) * u
        c, s = np.cos(th_arr[j]), np.sin(th_arr[j])
        u, v = c * ut - 1j * s * vt, c * vt - 1j * s * ut
        phis.append(phi)
        u_tilde.append(ut)
        v_tilde.append(vt)
        u_next.append(u)
        v_next.append(v)

    raw_p_plus = np.abs(v) ** 2
    p_plus = np.clip(raw_p_plus, 1e-15, 1.0 - 1e-15)
    pi_ = np.where(s_arr > 0, p_plus, 1.0 - p_plus)

    # 1. 加权 NLL 与半空间象限屏障
    nll = -np.sum(w * np.log(pi_))
    z = 14.0 * (0.56 - pi_)
    margin_pen = 0.5 * np.sum(w * np.logaddexp(0.0, z))

    # 2. 有效场防反转护栏与经典直积能量项
    m_vec = 2.0 * p_plus - 1.0
    h_eff = h_arr + J_arr @ m_vec
    zf = 10.0 * (s_arr * h_eff + 0.08)
    inv_pen = (lam_inv / 10.0) * float(np.sum(nz_mask * np.logaddexp(0.0, zf))) if lam_inv > 0.0 else 0.0
    e_prod = 0.5 * float(m_vec @ J_arr @ m_vec) + float(h_arr @ m_vec)
    energy_pen = alpha_e * e_prod if alpha_e != 0.0 else 0.0

    # 3. 锚点信任域正则
    reg = 0.5 * mu_g * float(np.sum((g_arr - g0_arr) ** 2)) + 0.5 * mu_th * float(np.sum((th_arr - mix0_arr) ** 2))
    loss = float(nll + margin_pen + inv_pen + energy_pen + reg)

    # 对 p_i^+ 的精确链式法则梯度
    sig_z = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    dL_dpi = w * (-1.0 / pi_ - 7.0 * sig_z)
    dL_dp_plus = np.where(s_arr > 0, dL_dpi, -dL_dpi)
    if lam_inv > 0.0:
        sig_zf = 1.0 / (1.0 + np.exp(-np.clip(zf, -40.0, 40.0)))
        r_u = lam_inv * nz_mask * s_arr * sig_zf
        dL_dp_plus = dL_dp_plus + 2.0 * (J_arr @ r_u)
    if alpha_e != 0.0:
        dL_dp_plus = dL_dp_plus + 2.0 * alpha_e * h_eff

    # 逆向 O(kn) SU(2) 伴随传播
    bu, bv = -0.5 * dL_dp_plus * u, 0.5 * dL_dp_plus * v
    grad_g = mu_g * (g_arr - g0_arr)
    grad_th = mu_th * (th_arr - mix0_arr)
    for j in range(k - 1, -1, -1):
        grad_th[j] += 2.0 * float(np.sum(np.imag(np.conj(bu) * v_next[j] + np.conj(bv) * u_next[j])))
        c, s = np.cos(th_arr[j]), np.sin(th_arr[j])
        but, bvt = c * bu + 1j * s * bv, c * bv + 1j * s * bu
        grad_g[j] += -4.0 * float(np.sum(h_arr * np.imag(np.conj(but) * u_tilde[j] - np.conj(bvt) * v_tilde[j])))
        bu, bv = phis[j] * bvt, np.conj(phis[j]) * but
    return loss, grad_g, grad_th


def refine_multiscalar_anchor(
    a_cand: float,
    norm_r: np.ndarray,
    mixers0: np.ndarray,
    J: np.ndarray,
    h: np.ndarray,
    s_target: np.ndarray,
    G: float = GAMMA_MAX,
    dg: float = 3.5,
    dth: float = 0.75,
    th_max: float = 1.40,
    lam_inv: float = 1.5,
    alpha_e: float = 1.5,
    mu_g: float = 0.005,
    mu_th: float = 0.03,
    weighted: bool = True,
) -> dict:
    """围绕单个一维平顶解析锚点 a_cand 执行分阶段多标量伴随 L-BFGS-B 精修。

    Step 1a：固定 Daubechies 平顶混合角 mixers0，48 步 L-BFGS-B 对齐 k 个代价角 g；
    Step 1b：当非零局域场个数 n_nz > k 且 dth > 0 时，在非退化混合角边界 [-th_max, th_max] 内
             释放全部 2k 个参数 (g, theta) 执行 64 步联合伴随精修。
    """
    norm_r_arr = np.asarray(norm_r, dtype=np.float64)
    mix0_arr = np.asarray(mixers0, dtype=np.float64)
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    s_arr = np.asarray(s_target, dtype=np.float64)

    k = len(norm_r_arr)
    nz = np.abs(h_arr) > 1e-8
    n_nz = int(np.sum(nz))
    g0 = float(a_cand) * norm_r_arr
    bnds_g = [(max(-G, float(g0[j]) - dg), min(G, float(g0[j]) + dg)) for j in range(k)]

    eff_lam = 0.0 if n_nz <= k else lam_inv
    eff_alpha = 0.0 if n_nz <= k else alpha_e

    if n_nz <= k or dth <= 0.0:
        res1 = minimize(
            lambda g: adjoint_echo_loss_and_grad(
                g, mix0_arr, h_arr, J_arr, s_arr, g0, mix0_arr,
                lam_inv=eff_lam, alpha_e=eff_alpha, mu_g=mu_g, mu_th=0.0, weighted=weighted,
            )[:2],
            g0,
            jac=True,
            method="L-BFGS-B",
            bounds=bnds_g,
            options={"maxiter": 48},
        )
        gc, thc = res1.x, mix0_arr.copy()
    else:
        res1 = minimize(
            lambda g: adjoint_echo_loss_and_grad(
                g, mix0_arr, h_arr, J_arr, s_arr, g0, mix0_arr,
                lam_inv=eff_lam, alpha_e=eff_alpha, mu_g=mu_g, mu_th=0.0, weighted=weighted,
            )[:2],
            g0,
            jac=True,
            method="L-BFGS-B",
            bounds=bnds_g,
            options={"maxiter": 48},
        )
        bnds_th = [
            (
                max(min(-th_max, float(mix0_arr[j])), float(mix0_arr[j]) - dth),
                min(max(th_max, float(mix0_arr[j])), float(mix0_arr[j]) + dth),
            )
            for j in range(k)
        ]

        def obj_joint(x: np.ndarray) -> tuple[float, np.ndarray]:
            l_val, gg, gth = adjoint_echo_loss_and_grad(
                x[:k], x[k:], h_arr, J_arr, s_arr, g0, mix0_arr,
                lam_inv=eff_lam, alpha_e=eff_alpha, mu_g=mu_g, mu_th=mu_th, weighted=weighted,
            )
            return l_val, np.r_[gg, gth]

        res2 = minimize(
            obj_joint, np.r_[res1.x, mix0_arr], jac=True, method="L-BFGS-B",
            bounds=bnds_g + bnds_th, options={"maxiter": 64},
        )
        res3 = minimize(
            obj_joint, np.r_[g0, mix0_arr], jac=True, method="L-BFGS-B",
            bounds=bnds_g + bnds_th, options={"maxiter": 64},
        )
        best_res = res2 if res2.fun <= res3.fun else res3
        gc, thc = best_res.x[:k], best_res.x[k:]

    minus, plus = spinors(gc, thc, h_arr)
    p_plus = np.abs(plus) ** 2
    pi_ = np.where(s_arr > 0, p_plus, np.abs(minus) ** 2)
    m_vec = 2.0 * p_plus - 1.0
    e_prod = float(0.5 * m_vec @ J_arr @ m_vec + h_arr @ m_vec)
    min_pi = float(np.min(pi_[nz])) if n_nz > 0 else 0.5
    clipped_pi = np.clip(pi_, 1e-15, 1.0)
    log_p = float(np.sum(np.log(clipped_pi)))
    mean_log_p_nz = float(np.mean(np.log(clipped_pi[nz]))) if n_nz > 0 else 0.0
    quadrant_hit = bool(min_pi > 0.50)
    c_quad, c_nll = 0.25, 0.05
    j_select = float(e_prod - (c_quad if quadrant_hit else 0.0) - c_nll * mean_log_p_nz)
    score = float(-j_select)
    return dict(
        anchor=float(a_cand),
        g=gc,
        th=thc,
        prob=float(np.exp(log_p)),
        log_p=log_p,
        e_prod=e_prod,
        min_pi=min_pi,
        quadrant_hit=quadrant_hit,
        j_select=j_select,
        score=score,
    )


def generate_stage1_angles(
    variant: str,
    J: np.ndarray,
    h: np.ndarray,
    pi: np.ndarray,
    depth: int,
    norm_r: np.ndarray | None = None,
    mixers0: np.ndarray | None = None,
    scale: float | None = None,
    G: float = GAMMA_MAX,
    is_online: bool = True,
) -> tuple[np.ndarray, dict]:
    """在指定 Stage-1 变体下生成 2p 维初始角度及 Top-1/Top-2 候选诊断信息（0 次量子评估）。"""
    if variant not in VARIANTS:
        raise ValueError(f"未知多标量变体：{variant}")
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    k = depth // 2

    if norm_r is None or mixers0 is None or scale is None:
        formula = load_formula("Flat-Top", depth)
        mixers0 = np.asarray(formula["mixers"], dtype=np.float64)
        ratios = np.asarray(formula["ratios"], dtype=np.float64)
        scale = float(np.max(np.abs(ratios)))
        norm_r = ratios / scale
    else:
        norm_r = np.asarray(norm_r, dtype=np.float64)
        mixers0 = np.asarray(mixers0, dtype=np.float64)
        scale = float(scale)

    # Echo-Flat-1D: 一维单标量平顶 Echo（在线 128 点重叠解码；离线 flat_scan 解析根区间扫描）
    if variant == "Echo-Flat-1D":
        if is_online:
            best_a, ang, overlap = decode("Flat-Top", depth, h_arr, pi_arr, grid_points=128)
            s_dir = np.where(pi_arr >= 0.5, 1.0, -1.0)
            m_, p_ = spinors(best_a * norm_r, mixers0, h_arr)
            p_plus = np.abs(p_) ** 2
            m_vec = 2.0 * p_plus - 1.0
            e_prod = float(0.5 * m_vec @ J_arr @ m_vec + h_arr @ m_vec)
            return ang, dict(
                variant=variant,
                s_target=s_dir,
                quadrant_hit=0,
                scalar=float(best_a),
                overlap=float(overlap),
                e_prod=e_prod,
                top1_angles=ang,
                top2_angles=ang.copy(),
            )
        s_target = 2.0 * pi_arr - 1.0
        ap, sp, _ = flat_scan(h_arr, s_target, k, G / scale)
        cands = list(start_batches(ap * scale, sp))[0]
        ang1 = expand_echo_angles(cands[0] * norm_r, mixers0, depth)
        ang2 = expand_echo_angles(cands[min(1, len(cands) - 1)] * norm_r, mixers0, depth)
        m_, p_ = spinors(cands[0] * norm_r, mixers0, h_arr)
        p_plus = np.abs(p_) ** 2
        pi_ = np.where(s_target > 0, p_plus, np.abs(m_) ** 2)
        m_vec = 2.0 * p_plus - 1.0
        e_prod = float(0.5 * m_vec @ J_arr @ m_vec + h_arr @ m_vec)
        nz = np.abs(h_arr) > 1e-8
        q_hit = int(np.min(pi_[nz]) > 0.50) if np.any(nz) else 0
        return ang1, dict(
            variant=variant,
            s_target=s_target,
            quadrant_hit=q_hit,
            scalar=float(cands[0]),
            e_prod=e_prod,
            top1_angles=ang1,
            top2_angles=ang2,
        )

    # 确定目标自旋方向并提取一维平顶解析峰锚点
    if variant == "Echo-Flat-kD (Cost-Only)" and is_online:
        s_target = np.where(pi_arr >= 0.5, 1.0, -1.0)
    else:
        s_target = resolve_prior_direction(J_arr, h_arr, pi_arr)

    ap, sp, _ = flat_scan(h_arr, s_target, k, G / scale)
    cands = list(start_batches(ap * scale, sp))[0][:10]

    variant_params = {
        "Echo-Flat-kD (Cost-Only)": dict(
            dg=3.5, dth=0.0, lam_inv=0.0, alpha_e=0.0, weighted=False,
        ),
        "Echo-Flat-kD (Guarded)": dict(
            dg=3.5, dth=0.0, lam_inv=1.5, alpha_e=1.5, weighted=True,
        ),
        "Echo-Flat-2kD (Unconstrained)": dict(
            dg=3.5, dth=np.pi, th_max=np.pi / 2.0, lam_inv=0.0, alpha_e=1.5, weighted=True,
        ),
        "Echo-Flat-2kD": dict(
            dg=3.5, dth=0.75, th_max=1.40, lam_inv=1.5, alpha_e=1.5,
            mu_g=0.005, mu_th=0.03, weighted=True,
        ),
    }[variant]

    refined = [
        refine_multiscalar_anchor(
            a, norm_r, mixers0, J_arr, h_arr, s_target, G, **variant_params
        )
        for a in cands
    ]
    refined.sort(key=lambda item: item["j_select"])
    best = refined[0]
    second = refined[min(1, len(refined) - 1)]
    ang1 = expand_echo_angles(best["g"], best["th"], depth)
    ang2 = expand_echo_angles(second["g"], second["th"], depth)
    return ang1, dict(
        variant=variant,
        s_target=s_target,
        quadrant_hit=int(best["quadrant_hit"]),
        e_prod=float(best["e_prod"]),
        j_select=float(best["j_select"]),
        prob=float(best["prob"]),
        top1_angles=ang1,
        top2_angles=ang2,
        candidates=refined,
    )
