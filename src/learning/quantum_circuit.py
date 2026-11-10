"""神经参数生成共用的角度损失与完整 QAOA 线路。"""
from __future__ import annotations

import numpy as np
import torch

from basis import flips_for, spin_table

GAMMA_MAX = 10 * np.pi
BETA_MAX = np.pi / 2


def _distance_values(prediction: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算整组反号下的 gamma 和 beta 周期距离。"""
    depth = prediction.shape[-1] // 2
    values = []
    for sign in (1.0, -1.0):
        gamma = (prediction[..., :depth] - sign * labels[..., :depth]) / GAMMA_MAX
        beta = (torch.remainder(prediction[..., depth:] - sign * labels[..., depth:] + BETA_MAX, np.pi) - BETA_MAX) / BETA_MAX
        values.append((gamma.square().sum(-1) + beta.square().sum(-1)) / (2 * depth))
    return values[0], values[1]


def nearest_squared(prediction: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """返回每道题在多标签、整组反号和 beta 周期下的最近归一化平方距离。"""
    first, second = _distance_values(prediction[:, None, :], labels)
    distance = torch.minimum(first, second).masked_fill(~mask, torch.inf)
    return distance.min(dim=1).values


def energies(J: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """计算任意 n 的完整 2^n 个配置能量。"""
    n = h.shape[1]
    spins = torch.as_tensor(spin_table(n), dtype=torch.float64, device=J.device)
    J, h = J.double(), h.double()
    return 0.5 * torch.einsum("si,bij,sj->bs", spins, J, spins) + h @ spins.T


def probability_distribution(angles: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    """使用 complex128 演化并返回完整测量分布。"""
    angles = angles.double()
    state_count = energy.shape[1]
    n = int(np.log2(state_count))
    depth = angles.shape[1] // 2
    state = torch.ones_like(energy, dtype=torch.complex128) / np.sqrt(state_count)
    flips = flips_for(n)
    for layer in range(depth):
        state = torch.exp(-1j * angles[:, layer, None] * energy) * state
        c = torch.cos(angles[:, depth + layer, None])
        s = torch.sin(angles[:, depth + layer, None])
        for flip in flips:
            index = torch.as_tensor(flip, dtype=torch.long, device=state.device)
            state = c * state - 1j * s * state[:, index]
    return state.abs().square()
