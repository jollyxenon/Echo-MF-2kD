"""阻尼经典平均场不动点迭代与热力学统计量计算。"""
from __future__ import annotations

import numpy as np

MEAN_FIELD_STEPS = 80
MEAN_FIELD_DAMPING = 0.5


def mean_field(J: np.ndarray, h: np.ndarray, beta: float) -> np.ndarray:
    """用定点迭代计算阻尼经典平均场磁化向量。"""
    m = np.zeros_like(h)
    for _ in range(MEAN_FIELD_STEPS):
        local = h + np.einsum("nij,nj->ni", J, m)
        updated = np.tanh(-beta * local)
        m = (1 - MEAN_FIELD_DAMPING) * m + MEAN_FIELD_DAMPING * updated
    return m


def mean_field_thermodynamics(J: np.ndarray, h: np.ndarray, m: np.ndarray,
                              beta: float) -> tuple[np.ndarray, ...]:
    """返回平均场能量、熵、自由能和磁化率。"""
    energy = 0.5 * np.einsum("ni,nij,nj->n", m, J, m) + np.einsum("ni,ni->n", m, h)
    positive = (1 + m) / 2
    negative = (1 - m) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(
            np.where(positive > 0, positive * np.log(np.maximum(positive, 1e-300)), 0)
            + np.where(negative > 0, negative * np.log(np.maximum(negative, 1e-300)), 0), axis=1
        )
    free = energy - entropy / beta
    susceptibility = beta * np.mean(1 - m ** 2, axis=1)
    return energy, entropy, free, susceptibility
