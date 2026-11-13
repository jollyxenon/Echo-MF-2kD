"""逐比特自旋边缘先验到 Echo 参数族的确定性重叠解码与分阶段多标量伴随精修。"""
from __future__ import annotations

import numpy as np

from echo.variants import load_formula
from echo.core import G as GAMMA_MAX
from learning.features import mean_field


def plus_probability(name: str, pairs: int, ah: np.ndarray) -> np.ndarray:
    """计算给定标量与局域场乘积时，Echo 末态取正自旋的概率。"""
    t = np.sin(2 * np.asarray(ah))
    if pairs == 1:
        # 单 Echo 对没有两倍 cost 比例，原始递推响应为 sin(4ah)。
        return (1 + np.sin(4 * np.asarray(ah))) / 2
    if name == "Flat-Top":
        q = (1 + t) / 2
        series = np.ones_like(q)
        coefficient = 1.0
        for r in range(1, pairs):
            coefficient *= (pairs - 1 + r) / r
            series += coefficient * (1 - q) ** r
        return q ** pairs * series
    if name == "Alternating-Phase":
        return (1 + np.sin(4 * pairs * np.asarray(ah))) / 2
    formula = load_formula(name, 2 * pairs)
    u = np.sin(formula["B"])
    return (1 + np.sin((2 * pairs - 1) * np.arcsin(np.clip(u * t, -1, 1)))) / 2


def response(name: str, pairs: int, a: np.ndarray, h: np.ndarray) -> np.ndarray:
    """返回每个解码标量对应的逐比特正自旋概率。"""
    scalar = np.atleast_1d(np.asarray(a, dtype=np.float64))
    fields = np.asarray(h, dtype=np.float64)
    return plus_probability(name, pairs, scalar[:, None] * fields[None, :])


def expected_success(name: str, pairs: int, a: np.ndarray, h: np.ndarray,
                     prior: np.ndarray) -> np.ndarray:
    """计算逐比特先验分布下，Echo 参数的期望目标态概率（分布重叠度 S(a)）。"""
    plus = response(name, pairs, a, h)
    pi = np.asarray(prior, dtype=np.float64)
    return np.prod(pi[None, :] * plus + (1 - pi[None, :]) * (1 - plus), axis=1)


def expand_scalar(name: str, depth: int, scalar: float) -> np.ndarray:
    """将一维标量 a 展开为标准 QAOA 的 2p 维完整角度向量（全部 gamma 后接全部 beta）。"""
    formula = load_formula(name, depth)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    normalized = ratios / np.max(np.abs(ratios))
    costs = float(scalar) * normalized
    angles = np.asarray(formula["mixers"], dtype=np.float64)
    expanded_gamma = []
    expanded_beta = []
    for cost, mixer in zip(costs, angles):
        expanded_gamma.extend((-cost, cost))
        expanded_beta.extend((np.pi / 2, mixer))
    return np.asarray(expanded_gamma + expanded_beta, dtype=np.float64)


def decode(name: str, depth: int, h: np.ndarray, prior: np.ndarray,
           grid_points: int = 128, chunk: int = 4096) -> tuple[float, np.ndarray, float]:
    """在指定 Echo 族的一维角度范围内扫描最大化分布重叠度 S(a)。"""
    if depth <= 0 or depth % 2:
        raise ValueError("Echo 解码要求正偶数深度")
    if grid_points < 2:
        raise ValueError("扫描点数至少为 2")
    pairs = depth // 2
    formula = load_formula(name, depth)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    normalized = ratios / np.max(np.abs(ratios))
    maximum = GAMMA_MAX / np.max(np.abs(normalized))
    grid = np.linspace(-maximum, maximum, grid_points)
    best_a, best_score = 0.0, -np.inf
    for start in range(0, len(grid), chunk):
        candidates = grid[start:start + chunk]
        scores = expected_success(name, pairs, candidates, h, prior)
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best_a, best_score = float(candidates[index]), float(scores[index])
    expanded = expand_scalar(name, depth, best_a)
    return best_a, expanded, float(best_score)


def decode_mean_field(name: str, depth: int, J: np.ndarray, h: np.ndarray,
                      beta: float = 2.0, grid_points: int = 128) -> tuple[float, np.ndarray, float, np.ndarray]:
    """免训练经典平均场回波初始化（MF-Echo）：80 步阻尼平均场边缘 + 128 点分布重叠度解码。"""
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    m_mf = mean_field(J_arr[None, ...], h_arr[None, ...], beta)[0]
    pi_mf = (1.0 + m_mf) / 2.0
    best_a, expanded, overlap = decode(name, depth, h_arr, pi_mf, grid_points=grid_points)
    return best_a, expanded, overlap, pi_mf


def decode_product_energy(name: str, depth: int, J: np.ndarray, h: np.ndarray,
                          grid_points: int = 128) -> tuple[float, np.ndarray, float]:
    """无自旋先验的一维 Echo 流形直积态解析能量选点（Echo-Energy）。"""
    if depth <= 0 or depth % 2:
        raise ValueError("Echo 解码要求正偶数深度")
    pairs = depth // 2
    formula = load_formula(name, depth)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    normalized = ratios / np.max(np.abs(ratios))
    maximum = GAMMA_MAX / np.max(np.abs(normalized))
    grid = np.linspace(-maximum, maximum, grid_points)
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    plus = response(name, pairs, grid, h_arr)
    m = 2.0 * plus - 1.0
    e_prod = 0.5 * np.einsum("gi,ij,gj->g", m, J_arr, m) + m @ h_arr
    best_idx = int(np.argmin(e_prod))
    best_a = float(grid[best_idx])
    expanded = expand_scalar(name, depth, best_a)
    return best_a, expanded, float(e_prod[best_idx])


def decode_multiscalar(
    depth: int,
    J: np.ndarray,
    h: np.ndarray,
    prior: np.ndarray,
    variant: str = "Echo-Flat-2kD",
    is_online: bool = True,
) -> tuple[np.ndarray, dict]:
    """分阶段多标量伴随精修解码（Echo-Flat-2kD，0 次在线量子评估）。"""
    if depth <= 0 or depth % 2:
        raise ValueError("Echo 解码要求正偶数深度")
    from echo.multiscalar import generate_stage1_angles

    return generate_stage1_angles(
        variant,
        np.asarray(J, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        np.asarray(prior, dtype=np.float64),
        depth,
        is_online=is_online,
    )


def decode_mean_field_multiscalar(
    depth: int,
    J: np.ndarray,
    h: np.ndarray,
    beta: float = 2.0,
    variant: str = "Echo-Flat-2kD",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """免训练经典平均场 + 分阶段多标量伴随精修（Echo-MF-2kD，0 次在线量子评估）。"""
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    m_mf = mean_field(J_arr[None, ...], h_arr[None, ...], beta)[0]
    pi_mf = (1.0 + m_mf) / 2.0
    expanded, meta = decode_multiscalar(depth, J_arr, h_arr, pi_mf, variant=variant, is_online=True)
    return expanded, meta, pi_mf


def target_probability(name: str, depth: int, a: float,
                       h: np.ndarray, s_star: np.ndarray) -> float:
    """计算候选 Echo 参数在已知目标态上的逐比特精确概率。"""
    pairs = depth // 2
    plus = plus_probability(name, pairs, a * np.asarray(h))
    factors = np.where(np.asarray(s_star) > 0, plus, 1 - plus)
    return float(np.prod(factors))
