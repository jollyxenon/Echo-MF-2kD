"""仅依赖 J,h 的能量优化；目标态只允许在调用结束后评分。"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from autograd import value_and_grad
import autograd.numpy as anp

from basis import spin_table, flips_for
from instances.benchmark import energy_table
from compare.autodiff import evolve
from compare.methods import bounds, DEFAULT_CONFIG

OPTIONS = dict(DEFAULT_CONFIG["lbfgsb"])


def expectation(angles, energy, flips):
    """完整状态向量的能量期望，支持反向自动微分。"""
    state = evolve(angles, energy, flips)
    return anp.real(anp.sum(anp.conj(state) * energy * state))


energy_gradient = value_and_grad(expectation)


class EnergyProblem:
    """缓存能量表；不接受目标态、最优能量或目标概率。"""

    def __init__(self, J, h):
        """由系数构造经典模拟需要的能量表和翻转索引。"""
        self.energy = energy_table(spin_table(len(h)), J, h, 0.0)
        self.flips = flips_for(len(h))

    def value(self, angles):
        """计算能量而不计算梯度。"""
        return float(expectation(angles, self.energy, self.flips))

    def optimize(self, initial):
        """释放全部角度；返回已评价点中能量最低的可行点及停止原因。"""
        started = time.perf_counter()
        initial = np.asarray(initial, dtype=float)
        best, best_energy, initial_energy = None, np.inf, None
        calls = 0

        def loss(angles):
            """记录历史最好已评价点，不读取答案。"""
            nonlocal best, best_energy, initial_energy, calls
            value, gradient = energy_gradient(angles, self.energy, self.flips)
            calls += 1
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError("能量或梯度非有限")
            if initial_energy is None:
                initial_energy = float(value)
            if value < best_energy:
                best, best_energy = np.array(angles), float(value)
            return float(value), np.asarray(gradient)

        result = minimize(loss, initial, jac=True, method="L-BFGS-B",
                          bounds=bounds(10 * np.pi, len(initial) // 2), options=OPTIONS)
        return dict(initial_angles=initial.tolist(), angles=best.tolist(),
                    initial_energy=initial_energy, energy=best_energy,
                    calls=calls, iterations=int(result.nit),
                    status="converged" if result.success else "optimizer_stopped",
                    message=str(result.message), seconds=time.perf_counter() - started)


def refine(J, h, initial):
    """部署接口：单次能量局部优化，不设墙钟上限。"""
    return EnergyProblem(J, h).optimize(initial)


def fast_dist(angles: np.ndarray, energy: np.ndarray, flips: list[np.ndarray]) -> np.ndarray:
    """纯 NumPy 状态向量演化，返回归一化计算基测量概率分布。"""
    p = len(angles) // 2
    gammas = angles[:p]
    betas = angles[p:]
    n = len(flips)
    state = np.full(len(energy), 1.0 / np.sqrt(len(energy)), dtype=np.complex128)
    phase_factors = np.exp(-1j * np.outer(gammas, energy))
    cos_b = np.cos(betas)
    sin_b = -1j * np.sin(betas)
    for layer in range(p):
        state = phase_factors[layer] * state
        cb, sb = cos_b[layer], sin_b[layer]
        for i in range(n):
            state = cb * state + sb * state[flips[i]]
    dist = state.real ** 2 + state.imag ** 2
    return dist / np.sum(dist)


def fast_energy_and_grad(
    angles: np.ndarray, energy: np.ndarray, flips: list[np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray]:
    """纯 NumPy 前向与反向伴随微分，返回 (期望能量, 2p 精确梯度, 测量概率分布)。"""
    p = len(angles) // 2
    gammas = angles[:p]
    betas = angles[p:]
    n = len(flips)
    state = np.full(len(energy), 1.0 / np.sqrt(len(energy)), dtype=np.complex128)
    phase_factors = np.exp(-1j * np.outer(gammas, energy))
    cos_b = np.cos(betas)
    sin_b = -1j * np.sin(betas)
    saved_after_cost = [None] * p
    saved_mixer = [[None] * n for _ in range(p)]
    for layer in range(p):
        state = phase_factors[layer] * state
        saved_after_cost[layer] = state
        cb, sb = cos_b[layer], sin_b[layer]
        for i in range(n):
            saved_mixer[layer][i] = state
            state = cb * state + sb * state[flips[i]]
    dist = state.real ** 2 + state.imag ** 2
    dist /= np.sum(dist)
    val = float(dist @ energy)
    adj = (energy * state).astype(np.complex128)
    grad_g = np.empty(p, dtype=np.float64)
    grad_b = np.empty(p, dtype=np.float64)
    for layer in range(p - 1, -1, -1):
        cb = cos_b[layer]
        sb_conj = 1j * np.sin(betas[layer])
        s_b = np.sin(betas[layer])
        gb = 0.0
        for i in range(n - 1, -1, -1):
            in_st = saved_mixer[layer][i]
            fl_in = in_st[flips[i]]
            d_st = -s_b * in_st - 1j * cb * fl_in
            gb += 2.0 * float(np.real(np.vdot(adj, d_st)))
            adj = cb * adj + sb_conj * adj[flips[i]]
        grad_b[layer] = gb
        st_c = saved_after_cost[layer]
        grad_g[layer] = 2.0 * float(np.real(np.vdot(adj, -1j * energy * st_c)))
        adj = np.conj(phase_factors[layer]) * adj
    return val, np.concatenate((grad_g, grad_b)), dist


_HADAMARD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _hadamard_basis(k: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 k 比特 Walsh-Hadamard 正交矩阵与横向场本征值向量 (k - 2 * popcount)。"""
    if k not in _HADAMARD_CACHE:
        h1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
        mat = h1
        for _ in range(k - 1):
            mat = np.kron(mat, h1)
        eig = np.array([k - 2 * bin(x).count("1") for x in range(1 << k)], dtype=np.float64)
        _HADAMARD_CACHE[k] = (mat, eig)
    return _HADAMARD_CACHE[k]


def _mixer_unitary(beta: float, h_mat: np.ndarray, eig: np.ndarray) -> np.ndarray:
    """在 Walsh-Hadamard 基下构造精确全局横向场矩阵 exp(-i * beta * sum_i X_i)。"""
    return (h_mat * np.exp(-1j * beta * eig)[None, :]) @ h_mat


def shifted_circuit_distributions(
    theta: np.ndarray, delta: float, energy: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """精确计算全部 4p 个单参数移位电路 (theta +/- delta * e_ell) 与当前电路的计算基概率分布。

    利用全局混合算符张量积分解 exp(-i * beta * sum X) = M_{n1}(beta) (x) M_{n2}(beta) 与前缀态复用，
    返回 (dists_4p, base_dist)，其中 dists_4p 前 2p 行为 +delta 移位，后 2p 行为 -delta 移位。
    """
    p = len(theta) // 2
    gammas, betas = theta[:p], theta[p:]
    n1 = n // 2
    n2 = n - n1
    d1, d2 = 1 << n1, 1 << n2
    h1, e1 = _hadamard_basis(n1)
    h2, e2 = _hadamard_basis(n2)
    dim_s = 1 << n

    base = np.full(dim_s, 1.0 / np.sqrt(dim_s), dtype=np.complex128)
    phases = np.exp(-1j * np.outer(gammas, energy))
    ph_plus = np.exp(-1j * delta * energy)
    ph_minus = np.conj(ph_plus)

    def apply_mixer(mat: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        return (m1 @ mat.reshape(mat.shape[0], d1, d2) @ m2).reshape(mat.shape[0], dim_s)

    # 按层追加每层新分叉出的 4 个移位态 [g+delta, g-delta, b+delta, b-delta]
    active = np.empty((4 * p, dim_s), dtype=np.complex128)
    n_act = 0
    for layer in range(p):
        b_val = float(betas[layer])
        m1 = _mixer_unitary(b_val, h1, e1)
        m2 = m1 if n1 == n2 else _mixer_unitary(b_val, h2, e2)
        m1p = _mixer_unitary(b_val + delta, h1, e1)
        m2p = m1p if n1 == n2 else _mixer_unitary(b_val + delta, h2, e2)
        m1m = _mixer_unitary(b_val - delta, h1, e1)
        m2m = m1m if n1 == n2 else _mixer_unitary(b_val - delta, h2, e2)

        if n_act > 0:
            active[:n_act] *= phases[layer]
            active[:n_act] = apply_mixer(active[:n_act], m1, m2)

        st_c = phases[layer] * base
        tri = apply_mixer(np.vstack((st_c, st_c * ph_plus, st_c * ph_minus)), m1, m2)
        base = tri[0]
        active[n_act:n_act + 2] = tri[1:3]
        active[n_act + 2] = apply_mixer(st_c[None, :], m1p, m2p)[0]
        active[n_act + 3] = apply_mixer(st_c[None, :], m1m, m2m)[0]
        n_act += 4

    ordered = active[np.r_[
        np.arange(0, 4 * p, 4), np.arange(2, 4 * p, 4),
        np.arange(1, 4 * p, 4), np.arange(3, 4 * p, 4),
    ]]
    dists = ordered.real ** 2 + ordered.imag ** 2
    dists /= dists.sum(axis=1, keepdims=True)
    base_dist = base.real ** 2 + base.imag ** 2
    base_dist /= base_dist.sum()
    return dists, base_dist


def shot_noise_adam_optimize(
    initial_angles: np.ndarray,
    energy: np.ndarray,
    flips: list[np.ndarray],
    shots: int | None,
    rng: np.random.Generator | None,
    steps: int = 64,
    delta: float = 0.10,
) -> dict[int, np.ndarray]:
    """有限步长两点移位电路投影抽样梯度与余弦退火 Adam 优化器（docs/module-settings.md §8.2）。

    - 在原生全局脉冲参数空间选取满足 Nyquist 无混叠条件 (delta < pi / (2n)) 的移位步长 delta = 0.10 rad；
    - 初始点执行 1 次电路能量测量；每步对全部 2*2p 个正负移位电路 (theta +/- delta * e_ell) 分别执行多项分布
      投影测量抽样构造中心差分梯度 (E_+ - E_-) / (2 * delta)，并对当前电路执行 1 次测量以跟踪历史最低观测能量
      （t 步累计电路评估次数为 1 + (4p + 1) * t；当 shots=None 时使用精确移位电路期望能量作同优化器无噪对照）；
    - Adam (beta1=0.9, beta2=0.999, eps=1e-8) 配合 64 步余弦退火 (lr_max=0.04 -> lr_min=0.002)，在 {16, 32, 64} 输出快照。
    """
    theta = np.array(initial_angles, dtype=np.float64)
    dim = len(theta)
    p = dim // 2
    n = len(flips)
    scale = 1.0 / (2.0 * delta)

    m_adam = np.zeros(dim, dtype=np.float64)
    v_adam = np.zeros(dim, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr_max, lr_min = 0.04, 0.002

    def measure_energy(dist: np.ndarray) -> float:
        if shots is None:
            return float(dist @ energy)
        return float((rng.multinomial(shots, dist) @ energy) / float(shots))

    dists, base_dist = shifted_circuit_distributions(theta, delta, energy, n)
    best_theta = theta.copy()
    best_obs_e = measure_energy(base_dist)
    snapshots: dict[int, np.ndarray] = {}

    for t in range(1, steps + 1):
        if shots is None:
            e_all = dists @ energy
        else:
            e_all = np.array([
                float((rng.multinomial(shots, dists[k]) @ energy) / float(shots))
                for k in range(2 * dim)
            ], dtype=np.float64)
        grad_est = scale * (e_all[:dim] - e_all[dim:])

        m_adam = beta1 * m_adam + (1.0 - beta1) * grad_est
        v_adam = beta2 * v_adam + (1.0 - beta2) * (grad_est ** 2)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t / steps))
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        theta[:p] = np.clip(theta[:p], -10.0 * np.pi, 10.0 * np.pi)
        theta[p:] = (theta[p:] + np.pi / 2.0) % np.pi - np.pi / 2.0

        dists, base_dist = shifted_circuit_distributions(theta, delta, energy, n)
        cur_e = measure_energy(base_dist)
        if cur_e <= best_obs_e:
            best_obs_e = cur_e
            best_theta = theta.copy()
        if t in (16, 32, 64) or t == steps:
            snapshots[t] = best_theta.copy()

    return snapshots
