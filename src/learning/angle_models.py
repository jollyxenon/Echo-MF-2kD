"""带结构特征的 MLP、GNN，以及 Echo 先验的直接回归对照。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from learning.quantum_circuit import BETA_MAX, GAMMA_MAX


def wrap_beta(beta: torch.Tensor) -> torch.Tensor:
    """把 mixer 角度折回长度为 pi 的代表区间。"""
    return torch.remainder(beta + BETA_MAX, np.pi) - BETA_MAX


def to_angles(raw: torch.Tensor, depth: int) -> torch.Tensor:
    """将未约束输出转换成物理域内的 gamma 和周期 beta。"""
    gamma = raw[:, :depth].clamp(-GAMMA_MAX, GAMMA_MAX)
    beta = wrap_beta(raw[:, depth:2 * depth])
    return torch.cat((gamma, beta), dim=1)


class MLP(nn.Module):
    """输入维度和输出深度均可配置的 GELU 多层感知机。"""

    def __init__(self, input_dim: int, depth: int, width: int = 256, rounds: int = 3):
        """建立直接角度回归网络。"""
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.GELU()]
        for _ in range(max(0, rounds - 1)):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        layers.append(nn.Linear(width, 2 * depth))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回未约束角度，物理域转换由 to_angles 完成。"""
        return self.net(x)


class GNN(nn.Module):
    """使用节点和边特征消息传递、全局均值读出的图网络。"""

    def __init__(self, n_qubits: int, node_dim: int, depth: int, width: int = 128,
                 rounds: int = 4):
        """建立与题目规模、QAOA 深度匹配的 GNN。"""
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.node = nn.Sequential(nn.Linear(node_dim, width), nn.GELU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 2, width), nn.GELU(),
                          nn.Linear(width, width), nn.GELU())
            for _ in range(rounds)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                          nn.Linear(width, width), nn.GELU())
            for _ in range(rounds)
        ])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2 * depth))

    def forward(self, node: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """只在真实耦合边上做消息传递后输出完整角度向量。"""
        batch, source, target = (J != 0).nonzero(as_tuple=True)
        edge = J[batch, source, target]
        edge_feature = torch.stack((edge, edge.abs()), dim=-1)
        z = self.node(node)
        for message, update in zip(self.messages, self.updates):
            pair = torch.cat((z[batch, source], z[batch, target], edge_feature), dim=-1)
            aggregate = z.new_zeros(len(J) * self.n_qubits, z.shape[-1])
            aggregate.index_add_(0, batch * self.n_qubits + source, message(pair))
            aggregate = aggregate.view(len(J), self.n_qubits, z.shape[-1]) / max(self.n_qubits - 1, 1)
            z = z + update(torch.cat((z, aggregate), dim=-1))
        return self.head(z.mean(1))
