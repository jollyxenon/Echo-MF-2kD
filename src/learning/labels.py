"""统一的六方法标签采集、完整线路复算和标签打包。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from basis import flips_for, spin_table
from cli.compare_echo import solve
from compare.autodiff import evolve
from hardware import choose_cpus
from instances.benchmark import energy_table
from learning.distance import ECHO_METHODS, METHODS, select
from records import (append_jsonl, check_snapshot, emit, load_json, save_json, save_npz,
                     snapshot_sources, source_files)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAP = 10.0
MAX_LABELS = 4
SEED = 0


class CandidateStream:
    """消费 Objective 事件，每次局部优化只保存其已评估最好点。"""

    def __init__(self):
        """初始化候选池和可观测事件。"""
        self.pool = []
        self.events = []
        self.local_id = 0
        self.active = False
        self.best = None
        self.evaluations = 0
        self.ineligible = 0

    def finish_local(self):
        """结束当前局部优化并保留其最好有效点。"""
        if self.best is not None:
            self.pool.append(self.best)
        self.best = None
        self.active = False

    def write(self, text: str):
        """接收 Objective 写入的 JSON 行。"""
        row = json.loads(text)
        kind = row["kind"]
        if kind == "local_start":
            self.finish_local()
            self.local_id += 1
            self.active = True
        if kind == "evaluation":
            self.evaluations += 1
            if not row["eligible"]:
                self.ineligible += 1
                return
            item = dict(angles=row["angles"], probability=row["probability"],
                        completed_at=row["completed_at"], stage=row["stage"],
                        source="local_best" if self.active else "evaluated_seed",
                        local_id=self.local_id if self.active else None)
            if self.active:
                if self.best is None or item["probability"] > self.best["probability"]:
                    self.best = item
            else:
                self.pool.append(item)
        else:
            self.events.append(row)
            if kind == "local_exit":
                self.finish_local()

    def flush(self):
        """实现文本流接口；事件已保存在内存中。"""


def protocol(n: int, depth: int, cap: float) -> dict:
    """返回当前题目规模和深度的标签协议。"""
    return dict(version=2, n=n, depth=depth, methods=list(METHODS), cap=cap,
                relative_floor=0.9, distance_threshold=0.01,
                maximum_labels=MAX_LABELS, seed=SEED)


def collect_one(data: dict[str, np.ndarray], index: int, method: str,
                depth: int, cap: float) -> dict:
    """为一道题执行一个方法，并独立复算所有有效候选。"""
    n = int(data["h"].shape[1])
    stream = CandidateStream()
    variant = method if method in ECHO_METHODS else None
    result, formula = solve(data["J"][index], data["h"][index], data["s_star"][index], depth,
                            variant, stream, cap, method, SEED + index + depth * 100000)
    stream.finish_local()
    if not stream.pool or result["status"] == "construction_over_cap":
        raise ValueError("预算内没有真实可用候选")
    started = time.perf_counter()
    spins = spin_table(n)
    energy = energy_table(spins, data["J"][index], data["h"][index], 0.0)
    flips = flips_for(n)
    residual, norm_error = 0.0, 0.0
    for item in stream.pool:
        angles = np.asarray(item["angles"], dtype=float)
        if item["completed_at"] > cap or not np.all(np.isfinite(angles)):
            raise ValueError("超时候选或非有限角度进入候选池")
        if np.any(np.abs(angles[:depth]) > 10 * np.pi + 1e-12) or np.any(np.abs(angles[depth:]) > np.pi / 2 + 1e-12):
            raise ValueError("候选角度越界")
        state = evolve(angles, energy, flips)
        probability = float(abs(state[int(data["target_index"][index])]) ** 2)
        if not np.isfinite(probability):
            raise ValueError("目标概率非有限")
        residual = max(residual, abs(probability - item["probability"]))
        norm_error = max(norm_error, abs(float(np.vdot(state, state).real) - 1.0))
        if residual > 1e-10 or norm_error > 1e-10:
            raise ValueError("独立完整线路复算失败")
    selected = select(stream.pool, relative_floor=0.9, limit=MAX_LABELS)
    return dict(
        id=str(data["ids"][index]), method=method, protocol=protocol(n, depth, cap),
        result=result, formula=formula, pool=stream.pool, selected=selected,
        events=stream.events, evaluations=stream.evaluations, ineligible=stream.ineligible,
        verification=dict(probability_residual=residual, norm_error=norm_error,
                          seconds=time.perf_counter() - started, candidates=len(stream.pool)),
        completed=True,
    )


def worker(dataset: Path, output: Path, shard: int, workers: int, cpu: int,
           depth: int, cap: float, limit: int | None = None) -> None:
    """在一个物理核上逐题原子提交六种方法的标签。"""
    os.sched_setaffinity(0, {cpu})
    data = dict(np.load(dataset, allow_pickle=False))
    if not np.all(np.isfinite(data["J"])) or not np.all(np.isfinite(data["h"])):
        raise ValueError("题库系数包含非有限值")
    if not np.all(np.isin(data["s_star"], (-1, 1))):
        raise ValueError("目标自旋必须取 ±1")
    if len(np.unique(data["ids"])) != len(data["ids"]):
        raise ValueError("题库 ID 必须唯一")
    count = len(data["ids"]) if limit is None else min(limit, len(data["ids"]))
    log_path = output / f"worker-{shard:02d}.jsonl"
    for index in range(shard, count, workers):
        folder = output / str(data["ids"][index])
        folder.mkdir(exist_ok=True)
        order = np.random.default_rng([SEED, index, depth]).permutation(METHODS)
        for method in order:
            method = str(method)
            path = folder / f"{method}.json"
            if path.exists():
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored["protocol"] != protocol(data["h"].shape[1], depth, cap) or not stored["completed"]:
                    raise ValueError(f"已有结果协议不匹配: {path}")
                continue
            started = time.perf_counter()
            try:
                result = collect_one(data, index, method, depth, cap)
                save_json(path, result)
                record = dict(index=index, id=str(data["ids"][index]), method=method,
                              status="completed", selected=len(result["selected"]),
                              candidates=len(result["pool"]), seconds=time.perf_counter() - started,
                              cpu=cpu)
            except Exception:
                record = dict(index=index, id=str(data["ids"][index]), method=method,
                              status="failed", seconds=time.perf_counter() - started,
                              cpu=cpu, error=traceback.format_exc())
                save_json(folder / f"{method}.error.json", record)
                append_jsonl(log_path, record)
                raise
            append_jsonl(log_path, record)
            emit(record)


def collect(dataset: Path, output: Path, depth: int, cap: float = DEFAULT_CAP,
            workers: int = 16, limit: int | None = None, resume: bool = False) -> None:
    """协调固定分片，支持预运行后续跑。"""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    with np.load(dataset, allow_pickle=False) as source:
        n = int(source["h"].shape[1])
        count = len(source["ids"])
    current_protocol = protocol(n, depth, cap)
    if not 1 <= workers <= 16:
        raise ValueError("标签采集最多使用 16 个物理核")
    if limit is not None and not 1 <= limit <= count:
        raise ValueError("limit 超出题库范围")
    files = source_files()
    if resume:
        manifest = load_json(output / "manifest.json")
        if manifest["protocol"] != current_protocol or dataset.read_bytes() != (output / "dataset.npz").read_bytes():
            raise ValueError("续跑题库或协议不同")
        check_snapshot(output, files)
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_sources(output, files)
        shutil.copy2(dataset, output / "dataset.npz")
        save_json(output / "manifest.json", dict(
            protocol=current_protocol, dataset=str(dataset), host=platform.node(),
            python=sys.version, numpy=np.__version__))
    cpus = choose_cpus(workers)
    launch = dict(protocol=current_protocol, pid=os.getpid(), host=platform.node(),
                  workers=workers, cpus=cpus, limit=limit, resume=resume, started=time.time())
    save_json(output / f"launch-{time.time_ns()}.json", launch)
    processes = []
    for shard, cpu in enumerate(cpus):
        command = [sys.executable, str(ROOT / "src/cli/label_learning.py"), "worker",
                   "--dataset", str(dataset), "--output", str(output), "--workers", str(workers),
                   "--shard", str(shard), "--cpu", str(cpu), "--depth", str(depth), "--cap", str(cap)]
        if limit is not None:
            command.extend(("--limit", str(limit)))
        processes.append(subprocess.Popen(command, cwd=ROOT))
    codes = [process.wait() for process in processes]
    status = dict(**launch, returncodes=codes, finished=time.time(),
                  successful=all(code == 0 for code in codes))
    save_json(output / ("pilot-finished.json" if limit is not None else "finished.json"), status)
    if not status["successful"]:
        raise RuntimeError(f"标签进程失败: {codes}")


def pack(dataset: Path, labels: Path, depth: int, cap: float = DEFAULT_CAP) -> None:
    """读取全部六方法结果，打包原始标签和共同质量区间标签。"""
    labels = Path(labels)
    with np.load(dataset, allow_pickle=False) as data:
        ids, splits = data["ids"], data["split"]
        n = int(data["h"].shape[1])
    count = len(ids)
    method_count = len(METHODS)
    angle_count = 2 * depth

    def blank() -> dict[str, np.ndarray]:
        """创建不填充虚假标签的固定尺寸包。"""
        return dict(ids=ids, methods=np.asarray(METHODS),
                    angles=np.zeros((count, method_count, MAX_LABELS, angle_count)),
                    probabilities=np.zeros((count, method_count, MAX_LABELS)),
                    mask=np.zeros((count, method_count, MAX_LABELS), dtype=bool))

    original, matched = blank(), blank()
    coverage = np.zeros(count, dtype=bool)
    bins = np.full(count, -1, dtype=int)

    def put(store, row: int, method_index: int, chosen: list[dict]) -> None:
        """写入实际存在的候选，不复制填充标签。"""
        for k, item in enumerate(chosen[:MAX_LABELS]):
            store["angles"][row, method_index, k] = item["angles"]
            store["probabilities"][row, method_index, k] = item["probability"]
            store["mask"][row, method_index, k] = True

    expected = protocol(n, depth, cap)
    for index, name in enumerate(ids):
        pools = []
        for method_index, method in enumerate(METHODS):
            row = load_json(labels / str(name) / f"{method}.json")
            if row["id"] != str(name) or row["method"] != method or row["protocol"] != expected or not row["completed"]:
                raise ValueError(f"标签元数据不一致: {name} {method}")
            put(original, index, method_index, row["selected"])
            pools.append(row["pool"])
        groups = []
        for pool in pools:
            grouped = {}
            for item in pool:
                grouped.setdefault(min(39, int(item["probability"] / 0.025)), []).append(item)
            groups.append(grouped)
        common = set.intersection(*(set(group) for group in groups))
        if common:
            quality_bin = max(common)
            choices = [select(group[quality_bin], relative_floor=0.0) for group in groups]
            length = min(map(len, choices))
            for method_index, chosen in enumerate(choices):
                put(matched, index, method_index, chosen[:length])
            coverage[index] = True
            bins[index] = quality_bin
    matched.update(coverage=coverage, probability_bin=bins)
    for name, arrays in (("labels.npz", original), ("matched_labels.npz", matched)):
        save_npz(labels / name, arrays)
    summary = dict(count=count, n=n, depth=depth, methods=list(METHODS),
                   label_counts={method: {str(k): int(np.sum(original["mask"][:, m].sum(-1) == k))
                                          for k in range(1, MAX_LABELS + 1)}
                                 for m, method in enumerate(METHODS)},
                   matched_coverage={split: dict(total=int(np.sum(splits == split)),
                                                 matched=int(np.sum(coverage & (splits == split))))
                                     for split in ("train", "validation", "test")})
    save_json(labels / "summary.json", summary)
    emit(summary)
