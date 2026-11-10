"""完整 QAOA 线路复算所需的计算基索引工具。"""
from __future__ import annotations

import numpy as np


def spin_table(n: int) -> np.ndarray:
    """枚举自旋；整数最高位对应第 0 个比特，0/1 对应 -1/+1。"""
    x = np.arange(1 << n)
    shifts = np.arange(n - 1, -1, -1)
    bits = (x[:, None] >> shifts[None, :]) & 1
    return 2.0 * bits - 1.0


def target_index(s_star: np.ndarray) -> int:
    """把目标自旋串编码成计算基索引。"""
    index = 0
    for i, spin in enumerate(s_star):
        if spin > 0:
            index |= 1 << (len(s_star) - 1 - i)
    return index


def target_indices(s_star: np.ndarray) -> np.ndarray:
    """批量把目标自旋串编码成计算基索引。"""
    spins = np.asarray(s_star)
    if spins.ndim != 2:
        raise ValueError("批量目标态必须是二维数组")
    powers = 1 << np.arange(spins.shape[1] - 1, -1, -1)
    return ((spins > 0).astype(np.int64) @ powers).astype(np.int64)


def flips_for(n: int) -> list[np.ndarray]:
    """返回各比特的计算基翻转索引。"""
    index = np.arange(1 << n)
    return [index ^ (1 << (n - 1 - j)) for j in range(n)]
