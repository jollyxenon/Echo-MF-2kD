"""对评测记录按题目实例执行统一的非参数百分位 Bootstrap 统计推断。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from records import read_jsonl


def interval(values: np.ndarray, replicates: int = 4000, seed: int = 0) -> tuple[float, np.ndarray]:
    """按题目实例执行非参数百分位 Bootstrap，返回 (均值, [2.5%, 97.5%])。"""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, len(arr), size=(replicates, len(arr)))].mean(axis=1)
    return float(arr.mean()), np.quantile(samples, [0.025, 0.975])


def bootstrap_ci(values: np.ndarray, replicates: int = 4000,
                 seed: int = 0) -> tuple[float, float, float]:
    """统一实例级百分位 Bootstrap 接口，返回 (mean, ci_low, ci_high)。"""
    mean_val, bounds = interval(values, replicates=replicates, seed=seed)
    return mean_val, float(bounds[0]), float(bounds[1])


def read_records(path: Path) -> tuple[list[str], np.ndarray]:
    """按题号读取逐题结果，返回概率、能量、时间与 Stage-2 调用数。"""
    records = sorted(read_jsonl(path), key=lambda row: row["id"])
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("评价记录存在重复题号")
    return ids, np.asarray([[
        r["initial_probability"], r["probability"], r["initial_energy"],
        r["energy"], r["total_seconds"], r["stage2_calls"],
    ] for r in records], dtype=np.float64)
