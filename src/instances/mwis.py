"""生成跨 p 复用的 MWIS 题库。"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from basis import spin_table
from instances.benchmark import energy_table, energy_table_precise, make_problem, verify_objective
from records import emit, save_json, save_npz, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "mwis"
TRAIN_SIZE = 10
DEFAULT_SIZES = (6, 8, 10, 12, 14)
DEFAULT_P_VALUES = (8, 10, 12)
DEFAULT_SPLITS = (4096, 512, 1024)
CROSS_SCALE_SPLITS = (0, 0, 1024)
DEFAULT_SEED = 0
MAX_ATTEMPTS = 200


def split_name(index: int, splits: tuple[int, int, int]) -> str:
    """按固定顺序返回题目的数据划分。"""
    train, validation, _ = splits
    if index < train:
        return "train"
    if index < train + validation:
        return "validation"
    return "test"


def canonical_key(weights: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """按顶点权重排序构造重编号核验键。"""
    order = np.argsort(weights)
    return np.concatenate((weights[order], matrix[np.ix_(order, order)].reshape(-1)))


def make_one(n: int, index: int, seed: int, spins: np.ndarray, split: str) -> tuple[dict, dict, np.ndarray]:
    """重采样一题直到得到唯一基态，并返回数组、记录和重编号键。"""
    for attempt in range(MAX_ATTEMPTS):
        row_seed = [seed, 4, n, index, attempt]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(row_seed)))
        problem = make_problem("mwis", n, rng)
        J, h, C = (problem[name] for name in ("J", "h", "C"))
        energy = energy_table(spins, J, h, C)
        objective_error = verify_objective("mwis", spins, energy, problem)
        scale = float(np.sqrt((np.sum(J ** 2) + np.sum(h ** 2)) / n))
        normalized = energy_table(spins, J / scale, h / scale, 0.0)
        normalization_error = float(np.max(np.abs(normalized - (energy - C) / scale)))
        if normalization_error > 1e-10:
            raise ValueError(f"MWIS n={n} 第 {index} 题归一化核验失败")
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(energy))))
        near = np.flatnonzero(energy - energy.min() <= tolerance)
        precise = energy_table_precise(spins[near], J, h, C)
        precise_tolerance = 64 * np.finfo(np.longdouble).eps * max(1.0, float(np.max(np.abs(energy))))
        ground = near[np.abs(precise - precise.min()) <= precise_tolerance]
        if len(ground) != 1:
            continue
        target = int(ground[0])
        excited = np.delete(energy, target)
        gap = float(excited.min() - precise.min())
        weights = np.asarray(problem["vertex_weights"], dtype=np.float64)
        arrays = dict(
            J=J / scale,
            h=h / scale,
            C=np.array(0.0),
            J_raw=J,
            h_raw=h,
            C_raw=np.array(C),
            scale=np.array(scale),
            vertex_weights=weights,
            penalty=np.array(problem["penalty"]),
            s_star=spins[target],
            target_index=np.array(target),
            ground_indices=ground,
            near_ground_indices=near,
            edges=problem["edges"],
            edge_weights=problem["edge_weights"],
        )
        record = dict(
            id=f"mwis_n{n:02d}_{index:05d}",
            n=n,
            index=index,
            split=split,
            seed=row_seed,
            attempts=attempt + 1,
            ground_count=int(len(ground)),
            near_ground_count=int(len(near)),
            gap_raw=gap,
            target_index=target,
            objective_conversion_error=float(objective_error),
            normalization_error=normalization_error,
        )
        return arrays, record, canonical_key(weights, J)
    raise RuntimeError(f"MWIS n={n} 第 {index} 题在 {MAX_ATTEMPTS} 次重采样内未得到唯一目标态")


def splits_for_size(n: int, splits: tuple[int, int, int] | None) -> tuple[int, int, int]:
    """返回指定规模 n 的 (train, validation, test) 划分：默认 n=10 为 (4096, 512, 1024)，跨规模 n∈{6,8,12,14} 为 (0, 0, 1024)。"""
    if splits is not None:
        return splits
    return DEFAULT_SPLITS if n == TRAIN_SIZE else CROSS_SCALE_SPLITS


def generate(output: Path, sizes: tuple[int, ...] = DEFAULT_SIZES,
             p_values: tuple[int, ...] = DEFAULT_P_VALUES,
             splits: tuple[int, int, int] | None = None,
             seed: int = DEFAULT_SEED) -> None:
    """生成以 n=10 为训练/验证/测试中心（5,632 题）、n∈{6,8,12,14} 为跨规模测试集（各 1,024 题）的 MWIS 题库。"""
    if splits is not None and (len(splits) != 3 or any(value < 0 for value in splits) or sum(splits) <= 0):
        raise ValueError("划分必须是三个非负整数（train validation test）且总题数大于 0")
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 2 for size in sizes):
        raise ValueError("规模必须是不重复且至少为 2 的整数")
    if not p_values or len(set(p_values)) != len(p_values) or any(depth <= 0 or depth % 2 for depth in p_values):
        raise ValueError("Echo 深度必须是互不重复的正偶数")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    summaries = []
    for n in sizes:
        if n < 2:
            raise ValueError("MWIS 规模必须至少为 2")
        size_splits = splits_for_size(n, splits)
        total = sum(size_splits)
        folder = output / f"n{n:02d}"
        folder.mkdir()
        spins = spin_table(n)
        rows = []
        records = []
        keys = []
        for index in range(total):
            split = split_name(index, size_splits)
            arrays, record, key = make_one(n, index, seed, spins, split)
            rows.append(arrays)
            records.append(record)
            keys.append(key)
            if (index + 1) % 256 == 0 or index + 1 == total:
                emit(dict(n=n, completed=index + 1, total=total))
        key_array = np.asarray(keys)
        if len(np.unique(key_array, axis=0)) != total:
            raise ValueError(f"MWIS n={n} 存在重编号等价题")
        arrays = {
            "J": np.asarray([row["J"] for row in rows]),
            "h": np.asarray([row["h"] for row in rows]),
            "C": np.asarray([row["C"] for row in rows]),
            "J_raw": np.asarray([row["J_raw"] for row in rows]),
            "h_raw": np.asarray([row["h_raw"] for row in rows]),
            "C_raw": np.asarray([row["C_raw"] for row in rows]),
            "scale": np.asarray([row["scale"] for row in rows]),
            "vertex_weights": np.asarray([row["vertex_weights"] for row in rows]),
            "penalty": np.asarray([row["penalty"] for row in rows]),
            "s_star": np.asarray([row["s_star"] for row in rows]),
            "target_index": np.asarray([row["target_index"] for row in rows]),
            "ground_indices": np.asarray([row["ground_indices"] for row in rows]),
            "near_ground_indices": np.concatenate([row["near_ground_indices"] for row in rows]),
            "near_ground_offsets": np.r_[0, np.cumsum([len(row["near_ground_indices"]) for row in rows])],
            "ids": np.asarray([row["id"] for row in records]),
            "split": np.asarray([row["split"] for row in records]),
        }
        save_npz(folder / "dataset.npz", arrays)
        write_jsonl(folder / "generation.jsonl", records)
        summary = dict(
            n=n,
            count=total,
            splits=dict(train=size_splits[0], validation=size_splits[1], test=size_splits[2]),
            p_values=list(p_values),
            master_seed=seed,
            all_targets_unique=all(row["ground_count"] == 1 for row in records),
            permutation_duplicates=0,
            max_objective_conversion_error=max(row["objective_conversion_error"] for row in records),
            max_normalization_error=max(row["normalization_error"] for row in records),
            min_gap_raw=min(row["gap_raw"] for row in records),
        )
        save_json(folder / "manifest.json", summary)
        summaries.append(summary)
    manifest = dict(
        dataset="mwis",
        train_size=TRAIN_SIZE,
        sizes=list(sizes),
        p_values=list(p_values),
        default_train_splits=dict(train=DEFAULT_SPLITS[0], validation=DEFAULT_SPLITS[1], test=DEFAULT_SPLITS[2]),
        cross_scale_splits=dict(train=CROSS_SCALE_SPLITS[0], validation=CROSS_SCALE_SPLITS[1], test=CROSS_SCALE_SPLITS[2]),
        count=sum(s["count"] for s in summaries),
        master_seed=seed,
        python=platform.python_version(),
        numpy=np.__version__,
        total_seconds=perf_counter() - started,
        sizes_summary=summaries,
    )
    save_json(output / "manifest.json", manifest)
    emit(manifest)


def main() -> None:
    """解析 MWIS 题库生成参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--p-values", type=int, nargs="+", default=list(DEFAULT_P_VALUES))
    parser.add_argument("--splits", type=int, nargs=3, default=None, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output, tuple(args.sizes), tuple(args.p_values),
             tuple(args.splits) if args.splits is not None else None, args.seed)


if __name__ == "__main__":
    main()
