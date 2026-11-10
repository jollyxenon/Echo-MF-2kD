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


def shot_noise_adam_optimize(
    initial_angles: np.ndarray,
    energy: np.ndarray,
    flips: list[np.ndarray],
    shots: int,
    rng: np.random.Generator,
    steps: int = 64,
) -> dict[int, np.ndarray]:
    """含噪投影测量抽样与参数移位 Adam 优化器（docs/module-settings.md §8.2）。

    - 每步执行 2*2p 次多项分布投影测量抽样 rng.multinomial(shots, dist, size=2*dim)，
      合成无偏参数移位梯度与有限采样散粒噪声；
    - Adam (beta1=0.9, beta2=0.999, eps=1e-8) 配合 64 步余弦退火 (lr_max=0.04 -> lr_min=0.002)；
    - 每步通过 1 次多项分布测量跟踪历史最低观测能量参数，并在 t in {16, 32, 64} 输出快照。
    """
    theta = np.array(initial_angles, dtype=np.float64)
    dim = len(theta)
    p = dim // 2

    m_adam = np.zeros(dim, dtype=np.float64)
    v_adam = np.zeros(dim, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr_max, lr_min = 0.04, 0.002

    _, grad_ex, dist = fast_energy_and_grad(theta, energy, flips)
    best_theta = theta.copy()
    best_noisy_e = float((rng.multinomial(shots, dist) @ energy) / float(shots))
    snapshots: dict[int, np.ndarray] = {}

    for t in range(1, steps + 1):
        draws = rng.multinomial(shots, dist, size=(2 * dim,))
        shot_noise = ((draws[:dim] - draws[dim:]) @ energy) / (2.0 * float(shots))
        noisy_grad = grad_ex + shot_noise

        m_adam = beta1 * m_adam + (1.0 - beta1) * noisy_grad
        v_adam = beta2 * v_adam + (1.0 - beta2) * (noisy_grad ** 2)
        m_hat = m_adam / (1.0 - beta1 ** t)
        v_hat = v_adam / (1.0 - beta2 ** t)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t / steps))
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        theta[:p] = np.clip(theta[:p], -10.0 * np.pi, 10.0 * np.pi)
        theta[p:] = (theta[p:] + np.pi / 2.0) % np.pi - np.pi / 2.0

        _, grad_ex, dist = fast_energy_and_grad(theta, energy, flips)
        cur_e = float((rng.multinomial(shots, dist) @ energy) / float(shots))
        if cur_e <= best_noisy_e:
            best_noisy_e = cur_e
            best_theta = theta.copy()
        if t in (16, 32, 64) or t == steps:
            snapshots[t] = best_theta.copy()

    return snapshots
