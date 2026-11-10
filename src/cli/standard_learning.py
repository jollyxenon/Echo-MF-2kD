"""神经参数生成矩阵：六类角度教师、逐比特先验与冻结后能量优化。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hardware import physical_cores
from learning.aggregation import aggregate
from learning.distance import ECHO_METHODS, METHODS
from records import (SAVE_EVERY, check_snapshot, emit, load_json, load_npz, save_json,
                     snapshot_sources, source_files)


def run(output: Path, resume: bool):
    """冻结源码；标签两个 16 核任务，其余 32 个单线程任务；失败明确中止。"""
    output = output.resolve()
    files = source_files() + sorted((ROOT / "data/mwis").glob("n*/dataset.npz"))
    if resume:
        check_snapshot(output, files)
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_sources(output, files)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    cpus = physical_cores()
    if len(cpus) < 32:
        raise ValueError("标准远程矩阵需要 32 个可用物理核")
    default = ROOT / ".pixi/envs/default/bin/python"
    learning = ROOT / ".pixi/envs/learning/bin/python"
    lib_dirs = [
        str(ROOT / ".pixi/envs/learning/lib"),
        str(ROOT / ".pixi/envs/default/lib"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               LD_LIBRARY_PATH=":".join(p for p in lib_dirs if p))
    combos = [(10, p) for p in (8, 10, 12)]

    jobs = []
    for n, depth in combos:
        tag = f"n{n:02d}-p{depth:02d}"
        for architecture in ("MLP", "GNN", "Echo-Prior"):
            methods = ECHO_METHODS if architecture == "Echo-Prior" else METHODS
            for method in methods:
                name = f"{tag}__{architecture}__{method}__seed0"
                jobs.append(dict(id=name, n=n, depth=depth, architecture=architecture,
                                 method=method, seed=0,
                                 model=f"models/{name}/model.pt",
                                 evaluation=f"evaluation/{name}.jsonl"))
    datasets = {}
    for n in sorted({n for n, _ in combos}):
        data = load_npz(ROOT / f"data/mwis/n{n:02d}/dataset.npz")
        split = data["split"]
        datasets[str(n)] = dict(
            ids=[str(value) for value in data["ids"][split == "test"]],
            splits={name: int(np.sum(split == name)) for name in ("train", "validation", "test")},
        )
    parallel_slots = min(len(combos), len(cpus) // 16)
    manifest = dict(
        pid=os.getpid(), started=time.time(), cpus=cpus[:16 * parallel_slots],
        label_workers_per_task=16, label_parallel_tasks=parallel_slots,
        model_parallel_tasks=min(len(jobs), len(cpus)),
        combos=combos, seeds=[0], prior_seeds=[0], methods=list(METHODS),
        label_objective="target_probability", label_seconds_per_instance_method=10,
        inference_grid_points=128, test_wall_clock_cap=None, checkpoint_interval=SAVE_EVERY,
        datasets=datasets, jobs=jobs,
    )
    save_json(output / "manifest.json", manifest)

    def command(name, argv, cores=None):
        """为每个子任务独立保存日志与退出状态。"""
        args = [str(x) for x in argv]
        if cores is not None:
            args = ["taskset", "-c", ",".join(map(str, cores))] + args
        start = time.time()
        with (logs / f"{name}.log").open("a") as log:
            code = subprocess.run(args, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        status = dict(name=name, args=args, started=start, finished=time.time(), returncode=code)
        save_json(logs / f"{name}.json", status)
        emit(status)
        if code:
            raise RuntimeError(f"任务失败：{name}，见 {logs / (name + '.log')}")

    def labels_slot(slot):
        """互不重叠的 16 个物理核。"""
        for n, depth in combos[slot::parallel_slots]:
            tag = f"n{n:02d}-p{depth:02d}"
            folder = output / "labels" / tag
            dataset = ROOT / f"data/mwis/n{n:02d}/dataset.npz"
            finished_path = folder / "finished.json"
            complete = finished_path.exists() and load_json(finished_path)["successful"]
            if not complete:
                args = [default, "src/cli/label_learning.py", "collect", "--dataset", dataset,
                        "--output", folder, "--depth", depth, "--cap", 10, "--workers", 16]
                if folder.exists():
                    args.append("--resume")
                command("labels-" + tag, args, cpus[16 * slot:16 * (slot + 1)])
            if not (folder / "summary.json").exists():
                command("pack-" + tag, [default, "src/cli/label_learning.py", "pack",
                        "--dataset", dataset, "--labels", folder, "--depth", depth, "--cap", 10])

    with ThreadPoolExecutor(max_workers=parallel_slots) as executor:
        list(executor.map(labels_slot, range(parallel_slots)))

    model_jobs = []
    for job in jobs:
        tag = f"n{job['n']:02d}-p{job['depth']:02d}"
        folder = output / "models" / job["id"]
        common = ["--dataset", ROOT / f"data/mwis/n{job['n']:02d}/dataset.npz",
                  "--labels", output / "labels" / tag / "labels.npz",
                  "--output", folder, "--threads", 1]
        if job["architecture"] == "Echo-Prior":
            args = [learning, "src/cli/train_echo_prior.py", *common, "--method", job["method"],
                    "--seed", job["seed"], "--split", "validation", "--grid-points", 4097]
        else:
            args = [learning, "src/cli/train_angle_models.py", *common,
                    "--architecture", job["architecture"], "--method", job["method"],
                    "--seed", job["seed"]]
        model_jobs.append((job["id"], args, folder, folder / "summary.json"))

    def execute(job):
        """完整结果跳过；未完成训练隔离后从头运行，避免冒充优化器续训。"""
        name, args, folder, finished = job
        if finished.exists():
            return
        if folder.exists():
            folder.rename(folder.with_name(folder.name + f".interrupted-{time.time_ns()}"))
        command(name, args)

    model_workers = min(len(model_jobs), len(cpus))
    with ThreadPoolExecutor(max_workers=model_workers) as executor:
        list(executor.map(execute, model_jobs))

    def execute_evaluation(job):
        """评测已完成则跳过，失败从头重跑；不触碰训练目录。"""
        name, args, finished = job
        if not finished.exists():
            command(name, args)

    evaluation_jobs = []
    evaluation_dir = output / "evaluation"
    for job in jobs:
        evaluation_jobs.append((
            "evaluate-" + job["id"],
            [learning, "src/cli/evaluate_learning.py",
             "--dataset", ROOT / f"data/mwis/n{job['n']:02d}/dataset.npz",
             "--models", output / "models" / job["id"], "--output", evaluation_dir,
             "--split", "test", "--threads", 1, "--grid-points", 128],
            evaluation_dir / f"{job['id']}.json",
        ))
    eval_workers = min(len(evaluation_jobs), len(cpus))
    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        list(executor.map(execute_evaluation, evaluation_jobs))
    groups = aggregate(output)
    save_json(output / "summary.json", dict(
        successful=True, finished=time.time(), models=len(model_jobs),
        model_evaluations=len(evaluation_jobs), aggregate_groups=len(groups),
    ))


def main():
    """远程正式矩阵仅在显式执行该入口时启动。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, args.resume)


if __name__ == "__main__":
    main()
