"""按物理核分片运行 data/benchmark 全部题目、指定偶数深度、三次重复的六方法比较。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cli.compare_echo import BASELINE_ROWS, ECHO_ROWS
from echo.core import validate_depth
from echo.variants import METHOD_NAMES
from hardware import choose_cpus
from records import append_jsonl, emit, save_json, snapshot_sources, source_files


def worker(output, shard, workers, cpu, depths, cap):
    """每个实例深度使用独立进程，失败保留并返回非零。"""
    os.sched_setaffinity(0, {cpu})
    tasks = [(p, depth, 0) for p in sorted((ROOT / "data/benchmark").glob("*/*.npz"))
             for depth in depths]
    random.Random(0).shuffle(tasks)
    failures = 0
    for instance, depth, seed in tasks[shard::workers]:
        destination = output / f"{instance.stem}__p{depth}__s{seed}"
        if (destination / "result.json").exists():
            continue
        cap_args = ["--unconstrained"] if cap is None else ["--cap", str(cap)]
        command = [sys.executable, str(ROOT / "src/cli/compare_echo.py"), "--instance", str(instance),
                   "--depth", str(depth), "--seed", str(seed), *cap_args,
                   "--output", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        record = dict(instance=instance.stem, depth=depth, seed=seed,
                      returncode=result.returncode, stderr=result.stderr, stdout=result.stdout,
                      time=datetime.now(timezone.utc).isoformat())
        append_jsonl(output / f"shard-{shard:02d}.jsonl", record)
        emit(record)
        failures += int(result.returncode != 0)
    return int(failures > 0)


def main():
    """选择不同物理核，保存真实源码与配置后启动并等待分片。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 8, 10, 12, 14],
                        help="要比较的正偶数层数，默认 6 8 10 12 14")
    parser.add_argument("--cap", type=float, default=30.0,
                        help="每行方法的墙钟上限秒数（传入 0 表示无截断自然收敛轨 budget=None）")
    parser.add_argument("--unconstrained", action="store_true",
                        help="启用无截断自然收敛轨（budget = None）")
    parser.add_argument("--resume", action="store_true",
                        help="在已有输出目录上续跑：重新快照源码、写续跑记录，并跳过已有 result.json 的任务")
    args = parser.parse_args()
    if args.cap < 0:
        raise ValueError("上限预算不能为负")
    cap = None if (args.unconstrained or args.cap == 0) else args.cap
    for depth in args.depths:
        validate_depth(depth)
    if len(set(args.depths)) != len(args.depths):
        raise ValueError("比较深度不得重复")
    output = args.output.resolve()
    if args.shard is not None:
        raise SystemExit(worker(output, args.shard, args.workers, args.cpu, args.depths, cap))
    if args.resume:
        if not (output / "manifest.json").exists():
            raise ValueError("续跑需要输出目录里已有的 manifest.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
    cpus = choose_cpus(args.workers)
    snapshot_sources(output, source_files() + sorted((ROOT / "docs").glob("*.md")))
    manifest = dict(
        tasks=len(list((ROOT / "data/benchmark").glob("*/*.npz"))) * len(args.depths),
        echo_rows=list(ECHO_ROWS), baseline_rows=list(BASELINE_ROWS),
        method_names={row: METHOD_NAMES.get(row, row) for row in (*ECHO_ROWS, *BASELINE_ROWS)},
        depths=args.depths, seeds=[0], cpus=cpus, resumed=args.resume,
        started_at=datetime.now(timezone.utc).isoformat(),
        gamma_bounds=[-10 * math.pi, 10 * math.pi],
        budget_cap_seconds=cap,
        timing="每行方法（含 Echo 构造）统一使用同一个上限预算，截断分析复用改进轨迹",
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_json(output / (f"resume-{stamp}.json" if args.resume else "manifest.json"), manifest)
    children = []
    cap_cli = ["--unconstrained"] if cap is None else ["--cap", str(cap)]
    for shard, cpu in enumerate(cpus):
        log = (output / f"worker-{shard:02d}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, __file__, "--output", str(output), "--workers", str(args.workers),
             "--shard", str(shard), "--cpu", str(cpu), "--depths", *map(str, args.depths), *cap_cli],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        )
        children.append((process, log))
    codes = []
    for process, log in children:
        codes.append(process.wait())
        log.close()
    save_json(output / "finished.json", dict(
        worker_returncodes=codes, finished_at=datetime.now(timezone.utc).isoformat()))
    raise SystemExit(int(any(codes)))


if __name__ == "__main__":
    main()
