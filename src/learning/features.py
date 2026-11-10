"""不枚举配置的图结构、节点和平均场特征。"""
from __future__ import annotations

import numpy as np

BETAS = (1.0, 2.0, 5.0)
MEAN_FIELD_STEPS = 80
MEAN_FIELD_DAMPING = 0.5


def mean_field(J: np.ndarray, h: np.ndarray, beta: float) -> np.ndarray:
    """用定点迭代计算平均场磁化。"""
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


def node_features(J: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """构造每个比特的局部图和平均场特征。"""
    edge = J != 0
    degree = edge.sum(2).astype(float)
    weighted_degree = np.abs(J).sum(2)
    signed_degree = J.sum(2)
    max_edge = np.abs(J).max(2)
    neighbor_field_sum = np.where(edge, h[:, None, :], 0.0).sum(2)
    neighbor_field_mean = np.divide(neighbor_field_sum, np.maximum(degree, 1))
    columns = [h, degree, weighted_degree, signed_degree, max_edge, np.abs(h),
               neighbor_field_sum, neighbor_field_mean]
    names = ["h", "degree", "weighted_degree", "signed_degree", "max_edge", "abs_h",
             "neighbor_field_sum", "neighbor_field_mean"]
    for beta in BETAS:
        magnetization = mean_field(J, h, beta)
        columns.append(magnetization)
        names.append(f"mean_field_m_beta{beta:g}")
    columns.append(np.abs(mean_field(J, h, BETAS[1])))
    names.append("abs_mean_field_m_beta2")
    return np.stack(columns, axis=2), names


def global_features(J: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """构造每道题的全局图和平均场特征。"""
    edge = J != 0
    n_edges = edge.sum((1, 2)).astype(float)
    values = np.where(edge, np.abs(J), np.nan)
    with np.errstate(invalid="ignore"):
        mean_abs_edge = np.nanmean(values, axis=(1, 2))
    mean_abs_edge = np.nan_to_num(mean_abs_edge)
    columns = [np.sqrt((J ** 2).sum((1, 2))), np.sqrt((h ** 2).sum(1)), n_edges,
               mean_abs_edge, h.mean(1), np.abs(h).mean(1), np.abs(h).max(1),
               np.abs(h).min(1), np.abs(h).std(1)]
    names = ["frobenius_J", "norm_h", "n_edges", "mean_abs_edge", "mean_h", "mean_abs_h",
             "max_abs_h", "min_abs_h", "std_abs_h"]
    for beta in BETAS:
        m = mean_field(J, h, beta)
        energy, entropy, free, susceptibility = mean_field_thermodynamics(J, h, m, beta)
        columns += [energy, entropy, free, susceptibility, np.abs(m).mean(1)]
        names += [f"mf_energy_beta{beta:g}", f"mf_entropy_beta{beta:g}",
                  f"mf_free_beta{beta:g}", f"mf_susceptibility_beta{beta:g}",
                  f"mf_abs_m_beta{beta:g}"]
    return np.stack(columns, axis=1), names


def build(J: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray | list[str]]:
    """同时返回 MLP 展平特征、GNN 节点特征和全局特征。"""
    node, node_names = node_features(J, h)
    global_, global_names = global_features(J, h)
    flat = np.concatenate((node.reshape(len(J), -1), global_), axis=1)
    return dict(node=node, node_names=node_names, global_=global_, global_names=global_names, flat=flat)
