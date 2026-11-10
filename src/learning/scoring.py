"""逐题评分与运行级汇总的唯一口径。"""
from __future__ import annotations

import numpy as np
import torch

from basis import flips_for, spin_table
from compare.autodiff import evolve
from instances.benchmark import energy_table
from learning.quantum_circuit import energies, probability_distribution


def distributions(energy: np.ndarray, flips: list[np.ndarray],
                  angles: list[np.ndarray]) -> np.ndarray:
    """用独立 autograd 线路计算一组角度的完整测量分布。"""
    return np.stack([np.abs(evolve(a, energy, flips)) ** 2 for a in angles])


def score_angles(energy: np.ndarray, flips: list[np.ndarray], target: int,
                 initial: np.ndarray, final: np.ndarray) -> dict:
    """单引擎评分：返回起点/终点的目标概率与期望能量。"""
    distribution = distributions(energy, flips, [np.asarray(initial, dtype=float),
                                                 np.asarray(final, dtype=float)])
    return dict(initial_probability=float(distribution[0, target]),
                probability=float(distribution[1, target]),
                initial_energy=float(distribution[0] @ energy),
                energy=float(distribution[1] @ energy))


def score_pair(J: np.ndarray, h: np.ndarray, target: int,
               initial: np.ndarray, final: np.ndarray) -> dict:
    """评测主路径评分：再以 Torch complex128 线路交叉核验分布与能量。"""
    pair = np.stack((np.asarray(initial, dtype=float), np.asarray(final, dtype=float)))
    energy = energy_table(spin_table(len(h)), J, h, 0.0)
    flips = flips_for(len(h))
    distribution = distributions(energy, flips, pair)
    score = dict(initial_probability=float(distribution[0, target]),
                 probability=float(distribution[1, target]),
                 initial_energy=float(distribution[0] @ energy),
                 energy=float(distribution[1] @ energy))
    Jt = torch.as_tensor(np.stack((J, J)), dtype=torch.float64)
    ht = torch.as_tensor(np.stack((h, h)), dtype=torch.float64)
    with torch.no_grad():
        et = energies(Jt, ht)
        qt = probability_distribution(torch.as_tensor(pair), et).numpy()
    distribution_residual = float(np.max(np.abs(qt - distribution)))
    energy_residual = float(np.max(np.abs(et.numpy() - energy)))
    norm_error = float(np.max(np.abs(distribution.sum(axis=1) - 1)))
    if max(distribution_residual, energy_residual, norm_error) > 1e-10:
        raise ValueError("独立完整线路核验失败")
    return dict(**score, distribution_residual=distribution_residual,
                energy_residual=energy_residual, norm_error=norm_error)


def summarize_records(records: list[dict]) -> dict:
    """按逐题记录汇总运行级指标；字段顺序固定。"""
    return dict(
        count=len(records),
        mean_initial_probability=float(np.mean([r["initial_probability"] for r in records])),
        median_probability=float(np.median([r["probability"] for r in records])),
        mean_probability=float(np.mean([r["probability"] for r in records])),
        mean_initial_energy=float(np.mean([r["initial_energy"] for r in records])),
        mean_energy=float(np.mean([r["energy"] for r in records])),
        mean_stage1_calls=float(np.mean([r["stage1_calls"] for r in records])),
        mean_stage2_calls=float(np.mean([r["stage2_calls"] for r in records])),
        mean_stage1_seconds=float(np.mean([r["stage1_seconds"] for r in records])),
        mean_stage2_seconds=float(np.mean([r["stage2_seconds"] for r in records])),
        mean_total_seconds=float(np.mean([r["total_seconds"] for r in records])),
        status_counts={status: sum(r["status"] == status for r in records)
                       for status in sorted({r["status"] for r in records})},
    )
