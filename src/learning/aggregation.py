"""按题平均模型种子，再对测试题执行统一的实例级百分位 Bootstrap。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from learning.distance import ECHO_METHODS, METHODS
from records import load_json, read_jsonl, save_json


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


def aggregate(root: str | Path, replicates: int = 4000, seed: int = 0) -> list[dict]:
    """从 manifest 与扁平评测记录汇总 36 个角度模型及 3 个逐比特先验模型。"""
    root = Path(root)
    jobs = load_json(root / "manifest.json")["jobs"]
    n_train = int(jobs[0]["n"])
    lookup = {(int(job["depth"]), str(job["architecture"]), str(job["method"]), int(job["seed"])):
              str(job["id"]) for job in jobs}
    summaries = []
    for depth in (8, 10, 12):
        for architecture in ("MLP", "GNN", "Echo-Prior"):
            methods = ECHO_METHODS if architecture == "Echo-Prior" else METHODS
            for method in methods:
                record_file = root / "evaluation" / f"{lookup[(depth, architecture, method, 0)]}.jsonl"
                ids, model = read_records(record_file)
                mean_prob, ci_bounds = interval(model[:, 1], replicates=replicates, seed=seed)
                summaries.append(dict(
                    n=n_train, depth=depth, architecture=architecture, method=method,
                    count=len(ids), seeds=[0],
                    mean_initial_probability=float(model[:, 0].mean()),
                    mean_probability=mean_prob,
                    ci95=ci_bounds.tolist(),
                    seed_means=[float(model[:, 1].mean())],
                    mean_initial_energy=float(model[:, 2].mean()),
                    mean_energy=float(model[:, 3].mean()),
                    mean_total_seconds=float(model[:, 4].mean()),
                    mean_calls=float(model[:, 5].mean()),
                ))
    save_json(root / "aggregate.json", dict(
        unit="instance_after_seed_average", bootstrap_replicates=replicates, seed=seed,
        interval="percentile_95", multiple_comparison_adjustment=False, groups=summaries,
    ))
    with (root / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    return summaries
