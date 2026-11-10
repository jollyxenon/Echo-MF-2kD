"""生成五类问题家族、n=6..16、每类每规模 50 道题，并逐题枚举核验唯一基态。"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from records import append_jsonl, emit, save_json, save_npz
OUTPUT = ROOT / "data" / "benchmark"
FAMILIES = ("dense_ising", "sparse_ising", "grid_ising", "pinned_maxcut", "mwis")
SIZES = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
INSTANCES_PER_CELL = 50
GRID_SHAPES = {
    6: (2, 3),
    7: (1, 7),
    8: (2, 4),
    9: (3, 3),
    10: (2, 5),
    11: (1, 11),
    12: (3, 4),
    13: (1, 13),
    14: (2, 7),
    15: (3, 5),
    16: (4, 4),
}
MASTER_SEED = 0
PIN_SPIN = 0
MAX_ATTEMPTS = 200


def sparse_edges(n: int, rng: np.random.Generator) -> np.ndarray:
    """保留环，再独立加边，得到期望平均度为 3 的连通图。"""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                edges.append((i, j))
            elif rng.random() < 1 / (n - 3):
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int64)


def grid_edges(n: int) -> tuple[np.ndarray, list[int]]:
    """按 GRID_SHAPES 铺设矩形网格；质数规模取单行条带。"""
    rows, cols = GRID_SHAPES[n]
    edges = []
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if col + 1 < cols:
                edges.append((i, i + 1))
            if row + 1 < rows:
                edges.append((i, i + cols))
    return np.asarray(edges, dtype=np.int64), [rows, cols]


def make_problem(family: str, n: int, rng: np.random.Generator) -> dict:
    """制作题目系数，二值变量约定 x=(s+1)/2。"""
    shape = None
    source_edges = np.empty((0, 2), dtype=np.int64)
    source_edge_weights = np.empty(0)
    pin_indices = np.empty(0, dtype=np.int64)
    pin_weights = np.empty(0)
    pinned_spin = -1
    if family == "dense_ising":
        edges = np.column_stack(np.triu_indices(n, 1))
    elif family == "grid_ising":
        edges, shape = grid_edges(n)
    elif family == "pinned_maxcut":
        # 钉扎：先取 n+1 个自旋的 MaxCut 图，固定 self 0 为 +1，消去一维。
        source_edges = sparse_edges(n + 1, rng)
        source_edge_weights = rng.uniform(0.5, 1.5, len(source_edges))
        inner, pins = [], []
        for (i, j), weight in zip(source_edges, source_edge_weights):
            if i == PIN_SPIN or j == PIN_SPIN:
                pins.append((int(j - 1) if i == PIN_SPIN else int(i - 1), float(weight)))
            else:
                inner.append((int(i - 1), int(j - 1), float(weight)))
        edges = np.asarray([(i, j) for i, j, _ in inner], dtype=np.int64)
        edge_weights = np.asarray([w for _, _, w in inner])
        pin_indices = np.asarray([i for i, _ in pins], dtype=np.int64)
        pin_weights = np.asarray([w for _, w in pins])
        pinned_spin = PIN_SPIN
    else:
        edges = sparse_edges(n, rng)
    J = np.zeros((n, n))
    h = np.zeros(n)
    C = 0.0
    vertex_weights = np.empty(0)
    penalty = None
    if family.endswith("ising"):
        edge_weights = rng.normal(size=len(edges))
        edge_weights /= np.sqrt(2 * np.sum(edge_weights**2) / n)
        h = rng.normal(size=n)
        h /= np.sqrt(np.mean(h**2))
        J[edges[:, 0], edges[:, 1]] = edge_weights
    elif family == "pinned_maxcut":
        # s_0=+1 后，内部边仍是耦合，钉扎边变成外场 h_j=w_0j/2。
        J[edges[:, 0], edges[:, 1]] = edge_weights / 2
        np.add.at(h, pin_indices, pin_weights / 2)
        C = -float(source_edge_weights.sum()) / 2
    else:
        vertex_weights = rng.uniform(0.5, 1.5, n)
        penalty = 2 * float(vertex_weights.max())
        edge_weights = np.full(len(edges), penalty)
        J[edges[:, 0], edges[:, 1]] = penalty / 4
        h = -vertex_weights / 2
        np.add.at(h, edges[:, 0], penalty / 4)
        np.add.at(h, edges[:, 1], penalty / 4)
        C = -float(vertex_weights.sum()) / 2 + penalty * len(edges) / 4
    J += J.T
    return dict(J=J, h=h, C=C, edges=edges, edge_weights=edge_weights,
                vertex_weights=vertex_weights, penalty=penalty, shape=shape,
                source_edges=source_edges, source_edge_weights=source_edge_weights,
                pin_indices=pin_indices, pin_weights=pin_weights, pinned_spin=pinned_spin)


def all_spins(n: int) -> np.ndarray:
    """按生成器位序枚举，整数最高位对应第 0 个自旋。"""
    indices = np.arange(1 << n, dtype=np.uint32)
    bits = (indices[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (2 * bits.astype(np.int8) - 1)


def energy_table(S: np.ndarray, J: np.ndarray, h: np.ndarray, C: float) -> np.ndarray:
    """逐项累加能量，避免构造每个状态的 n×n 外积。"""
    E = np.full(len(S), C, dtype=np.result_type(J.dtype, h.dtype, np.float64))
    for i in range(len(h)):
        E += h[i] * S[:, i]
        for j in range(i + 1, len(h)):
            if J[i, j] != 0:
                E += J[i, j] * (S[:, i] * S[:, j])
    return E


def energy_table_precise(S, J, h, C):
    """用扩展精度逐项复核数值近基态；不依赖 float64 能量排序。"""
    return energy_table(S, J.astype(np.longdouble), h.astype(np.longdouble), np.longdouble(C))


def verify_objective(family, S, E, problem):
    """对应用题逐状态核对原始目标，返回最大转换误差。"""
    if family.endswith("ising"):
        return 0.0
    x = (S + 1) // 2
    objective = np.zeros(len(S))
    if family == "pinned_maxcut":
        for (i, j), weight in zip(problem["edges"], problem["edge_weights"]):
            objective -= weight * (x[:, i] != x[:, j])
        for j, weight in zip(problem["pin_indices"], problem["pin_weights"]):
            objective -= weight * (1 - x[:, j])
        error = float(np.max(np.abs(E - objective)))
        assert error < 1e-9, (family, error)
        return error
    for i, weight in enumerate(problem["vertex_weights"]):
        objective -= weight * x[:, i]
    for i, j in problem["edges"]:
        objective += problem["penalty"] * (x[:, i] * x[:, j])
    feasible = np.ones(len(S), dtype=bool)
    for i, j in problem["edges"]:
        feasible &= (x[:, i] * x[:, j]) == 0
    assert abs(E.min() - objective[feasible].min()) < 1e-9
    error = float(np.max(np.abs(E - objective)))
    assert error < 1e-9, (family, error)
    return error


def make_instance(family: str, family_id: int, n: int, instance: int, S: np.ndarray,
                  master_seed: int = MASTER_SEED):
    """重采样直到目标态唯一，返回待存储数组与元数据字段。"""
    for attempt in range(MAX_ATTEMPTS):
        seed = [master_seed, family_id, n, instance, attempt]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        problem = make_problem(family, n, rng)
        J, h, C = problem["J"], problem["h"], problem["C"]
        E = energy_table(S, J, h, C)
        error = verify_objective(family, S, E, problem)
        scale = float(np.sqrt((np.sum(J**2) + np.sum(h**2)) / n))
        normalized = energy_table(S, J / scale, h / scale, 0.0)
        scale_error = float(np.max(np.abs(normalized - (E - C) / scale)))
        assert scale_error < 1e-9
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(E))))
        near = np.flatnonzero(E - E.min() <= tolerance)
        precise = energy_table_precise(S[near], J, h, C)
        precision_tol = 64 * np.finfo(np.longdouble).eps * max(1.0, float(np.max(np.abs(E))))
        ground = near[np.abs(precise - precise.min()) <= precision_tol]
        assert np.allclose(J, J.T) and np.all(np.diag(J) == 0)
        assert np.isclose((np.sum((J / scale) ** 2) + np.sum((h / scale) ** 2)) / n, 1)
        if len(ground) != 1:
            continue
        target = int(ground[0])
        excited = np.delete(E, ground)
        next_energy = float(excited.min())
        arrays = dict(J=J / scale, h=h / scale, C=0.0, J_raw=J, h_raw=h, C_raw=C, scale=scale,
                      s_star=S[target], target_index=target, ground_indices=ground,
                      near_ground_indices=near, edges=problem["edges"],
                      edge_weights=problem["edge_weights"], vertex_weights=problem["vertex_weights"],
                      pinned_spin=problem["pinned_spin"], source_edges=problem["source_edges"],
                      source_edge_weights=problem["source_edge_weights"])
        row = dict(family=family, n=n, instance=instance, seed=seed, attempts=attempt + 1,
                   edges=len(problem["edges"]), average_degree=2 * len(problem["edges"]) / n,
                   grid_shape=problem["shape"], penalty=problem["penalty"],
                   pinned_spin=problem["pinned_spin"],
                   source_n=(n + 1 if problem["pinned_spin"] == PIN_SPIN else n),
                   source_edges=len(problem["source_edges"]), scale=scale, target_index=target,
                   s_star=S[target].tolist(), energy_min_raw=float(precise.min()),
                   next_energy_raw=next_energy, gap_raw=next_energy - float(precise.min()),
                   energy_min_normalized=(float(precise.min()) - C) / scale,
                   ground_count=len(ground), near_ground_count=len(near),
                   tolerance_raw=tolerance,
                   objective_optimum=(-float(precise.min()) if not family.endswith("ising") else None),
                   objective_conversion_max_error=error, normalization_max_error=scale_error)
        return arrays, row
    raise RuntimeError(f"{family} n={n} instance={instance} 在 {MAX_ATTEMPTS} 次重采样内未得到唯一目标态")


def generate(output: Path = OUTPUT, sizes: tuple[int, ...] = SIZES,
             instances_per_cell: int = INSTANCES_PER_CELL, master_seed: int = MASTER_SEED) -> None:
    """生成指定规模和数量的 Benchmark 题库。"""
    if not sizes or len(set(sizes)) != len(sizes) or any(size not in GRID_SHAPES for size in sizes):
        raise ValueError(f"规模必须来自 {tuple(GRID_SHAPES)}")
    if instances_per_cell <= 0:
        raise ValueError("每个家族和规模至少生成一道题")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    start = perf_counter()
    for n in sizes:
        S = all_spins(n)
        for family_id, family in enumerate(FAMILIES):
            for instance in range(instances_per_cell):
                tick = perf_counter()
                arrays, row = make_instance(family, family_id, n, instance, S, master_seed)
                name = f"{family}_n{n:02d}_{instance:02d}"
                folder = output / family
                folder.mkdir(exist_ok=True)
                save_npz(folder / f"{name}.npz", arrays)
                record = dict(id=name, path=f"{family}/{name}.npz",
                              seconds=perf_counter() - tick, **row)
                save_json(folder / f"{name}.json", record)
                append_jsonl(output / "generation.jsonl", record)
                records.append(record)
                emit(dict(id=name, ground_count=record["ground_count"],
                          gap_raw=round(record["gap_raw"], 6),
                          seconds=round(record["seconds"], 3)))
    fields = ["id", "family", "n", "instance", "seed", "attempts", "path", "edges", "average_degree",
              "grid_shape", "penalty", "pinned_spin", "source_n", "source_edges", "scale",
              "target_index", "ground_count", "near_ground_count", "gap_raw", "objective_optimum",
              "objective_conversion_max_error", "normalization_max_error", "seconds"]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = dict(count=len(records), sizes=list(sizes), families=list(FAMILIES),
                   instances_per_cell=instances_per_cell, master_seed=master_seed,
                   grid_shapes={str(k): list(v) for k, v in GRID_SHAPES.items()}, pinned_spin=PIN_SPIN,
                   python=platform.python_version(), numpy=np.__version__,
                   platform=platform.platform(), longdouble_bits=np.finfo(np.longdouble).bits,
                   total_seconds=perf_counter() - start,
                   max_objective_conversion_error=max(r["objective_conversion_max_error"] for r in records),
                   max_normalization_error=max(r["normalization_max_error"] for r in records),
                   max_attempts=max(r["attempts"] for r in records),
                   min_gap_raw=min(r["gap_raw"] for r in records),
                   all_targets_unique=all(r["ground_count"] == 1 for r in records))
    save_json(output / "summary.json", summary)
    emit(summary)


def main() -> None:
    """解析 Benchmark 题库生成参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--instances-per-cell", type=int, default=INSTANCES_PER_CELL)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    args = parser.parse_args()
    generate(args.output, tuple(args.sizes), args.instances_per_cell, args.seed)


if __name__ == "__main__":
    main()
